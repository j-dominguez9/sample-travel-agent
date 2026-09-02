"""Online evaluation: score live production spans and write the results back
onto the trace as Phoenix annotations.

This is the half of the system that makes evals *observability* rather than
testing. The offline path (`runner.py`) scores a fixed golden dataset and
answers "did this change help?". This path scores whatever real users actually
sent and answers "is it healthy right now?" — which is what the customer asked
for with "live eval in production, alerts in prod".

The two paths share one registry, so an evaluator is written once. What decides
whether it can run here is `online`: an evaluator that needs a ground-truth
label cannot run against live traffic, because production has no labels. That
is the real dividing line between the tiers — not just "code can't score taste",
but "most of the deterministic suite has nothing to compare against out here."
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from phoenix.client import Client

from evals.core.registry import REGISTRY


def _attr(row: Any, *names: str) -> Any:
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
    return None


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def reconstruct(spans: pd.DataFrame, agent_span_name: str = "travel_agent") -> list[dict]:
    """Rebuild `{"input": ..., "output": {"reply", "tool_calls"}}` per trace.

    The experiment path gets this shape for free because the task returns it.
    Here it has to be recovered from the span tree: the agent span carries the
    request and the reply, and each TOOL child carries one call's input and
    output. Same shape either way, so the same evaluators apply unchanged.
    """
    if spans.empty:
        return []

    records: list[dict] = []
    skipped: list[str] = []

    agents = spans[spans["name"] == agent_span_name]
    for _, agent in agents.iterrows():
        span_id = agent["context.span_id"]
        tool_calls = []
        for _, child in spans[spans["parent_id"] == span_id].iterrows():
            if str(child.get("span_kind")) != "TOOL":
                continue
            tool_calls.append({
                "name": child["name"],
                "input": _parse(_attr(child, "attributes.input.value")),
                "output": _parse(_attr(child, "attributes.output.value")),
            })
        request = _attr(agent, "attributes.input.value") or ""
        reply = _attr(agent, "attributes.output.value") or ""
        if not str(request).strip() or not str(reply).strip():
            # Spans emitted before the agent span carried input/output. They are
            # unevaluable, not failing — scoring them would report a judge's
            # "cannot evaluate" as a zero and drag every average down.
            skipped.append(span_id)
            continue
        records.append({
            "span_id": span_id,
            "trace_id": agent["context.trace_id"],
            "input": {"message": request},
            "output": {
                "reply": reply,
                "tool_calls": tool_calls,
            },
        })
    if skipped:
        print(f"  skipped {len(skipped)} span(s) with no recorded input/output")
    return records


def score(records: Iterable[dict], agent: str) -> pd.DataFrame:
    """Run every online-capable evaluator over the reconstructed records."""
    rows = []
    for reg in REGISTRY.for_agent(agent, online=True):
        for rec in records:
            if reg.applies is not None and not reg.applies(rec):
                continue  # not meaningful for this turn
            try:
                result = reg.evaluator.evaluate(
                    {"input": rec["input"], "output": rec["output"]}
                )[0]
            except Exception as exc:  # noqa: BLE001 — see below
                # Deliberately broad: a judge can fail on a rate limit, a
                # malformed span, or a model refusal, and one bad span must not
                # abort a monitoring sweep over hundreds. The error is recorded
                # as an annotation rather than dropped, so it stays visible —
                # that is the difference between this and swallowing it.
                rows.append({
                    "span_id": rec["span_id"], "annotation_name": reg.name,
                    "score": None, "label": "error",
                    "explanation": f"{type(exc).__name__}: {exc}",
                    "annotator_kind": "CODE" if reg.kind == "code" else "LLM",
                })
                continue
            rows.append({
                "span_id": rec["span_id"],
                "annotation_name": reg.name,
                "score": float(result.score) if result.score is not None else None,
                "label": result.label,
                "explanation": result.explanation,
                "annotator_kind": "CODE" if reg.kind == "code" else "LLM",
                "direction": reg.direction,
            })
    return pd.DataFrame(rows)


def purge(
    agent: str,
    project: str,
    *,
    since_minutes: int = 60 * 24 * 30,
    client: Client | None = None,
) -> dict[str, Any]:
    """Remove this agent's evaluator annotations from a project, within a window.

    Annotations are keyed by (span, name), so re-running a sweep updates in
    place but cannot retract a verdict that no longer applies — if an evaluator
    gains an applicability rule, its old scores linger on turns it should never
    have graded. Purging before a re-sweep is what keeps the platform showing
    what the current code would actually produce.

    Scoped two ways: by evaluator name, so human annotations (`user_feedback`)
    and other agents' scores are untouched; and by time window, because Phoenix
    rejects an unbounded delete unless you pass `delete_all`. Taking the bounded
    path deliberately — an eval harness should not hold a loaded gun.

    Raises on any failure rather than returning it, because an earlier version
    collected errors into a dict the caller never inspected and reported a
    no-op purge as a success.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    base = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")
    end = datetime.now(UTC) + timedelta(minutes=1)
    start = end - timedelta(minutes=since_minutes)

    deleted: dict[str, Any] = {}
    for reg in REGISTRY.for_agent(agent):
        q = urllib.parse.urlencode({
            "name": reg.name,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
        })
        url = f"{base}/v1/projects/{urllib.parse.quote(project)}/span_annotations?{q}"
        req = urllib.request.Request(url, method="DELETE")
        try:
            with urllib.request.urlopen(req) as resp:
                resp.read()
            deleted[reg.name] = "ok"
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"purge failed for {reg.name}: HTTP {exc.code} {exc.read().decode()[:200]}"
            ) from None
    return deleted


def read_annotations(
    *,
    project: str,
    agent_span_name: str = "travel_agent",
    since_minutes: int = 60,
    limit: int = 200,
    client: Client | None = None,
) -> pd.DataFrame:
    """Re-read a sweep's scores from Phoenix, in the shape `score()` returns.

    Downstream tasks need the scores but must not recompute them: the judges
    cost real money per call. Phoenix already holds them as annotations, so it
    is both the durable record and the hand-off between tasks — which also keeps
    each task independently re-runnable without a large XCom payload.
    """
    client = client or Client()
    start = datetime.now(UTC) - timedelta(minutes=since_minutes)
    spans = client.spans.get_spans_dataframe(
        project_identifier=project, start_time=start, limit=limit
    )
    if spans.empty:
        return pd.DataFrame()
    span_ids = spans[spans["name"] == agent_span_name]["context.span_id"].tolist()
    if not span_ids:
        return pd.DataFrame()

    ann = client.spans.get_span_annotations_dataframe(
        span_ids=span_ids, project_identifier=project
    )
    if ann.empty:
        return pd.DataFrame()
    out = ann.reset_index().rename(
        columns={"result.score": "score", "result.label": "label",
                 "result.explanation": "explanation"}
    )
    keep = [c for c in ("span_id", "annotation_name", "score", "label", "explanation")
            if c in out.columns]
    return out[keep]


def sweep(
    *,
    agent: str,
    project: str,
    since_minutes: int = 60,
    limit: int = 100,
    agent_span_name: str = "travel_agent",
    dry_run: bool = False,
    client: Client | None = None,
) -> pd.DataFrame:
    """Sample recent spans, evaluate them, and annotate them in place."""
    client = client or Client()
    start = datetime.now(UTC) - timedelta(minutes=since_minutes)
    spans = client.spans.get_spans_dataframe(
        project_identifier=project, start_time=start, limit=limit
    )
    records = reconstruct(spans, agent_span_name=agent_span_name)
    if not records:
        return pd.DataFrame()

    results = score(records, agent)
    if not dry_run and not results.empty:
        client.spans.log_span_annotations_dataframe(
            dataframe=results.dropna(subset=["score"]).set_index("span_id"),
            sync=True,
        )
    return results
