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
import random
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


def _nested(row: Any, namespace: str, key: str) -> Any:
    """Read `<namespace>.<key>` from a span, however Phoenix chose to shape it.

    Phoenix collapses a dotted attribute name into a nested object on the way
    into the dataframe, so `travel_agent.turn_index` arrives as the column
    `attributes.travel_agent` holding `{"turn_index": 1}` — not as the flat
    column the name suggests. Attributes it knows as semantic conventions
    (`session.id`) stay flat instead, so both shapes occur side by side on the
    same span and reading only one silently yields None.
    """
    flat = _attr(row, f"attributes.{namespace}.{key}")
    if flat is not None:
        return flat
    ns = _attr(row, f"attributes.{namespace}")
    if isinstance(ns, dict):
        return ns.get(key)
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
        # Session identity travels inside `input` rather than beside it: the
        # evaluators are bound to `{"input", "output"}` and the offline
        # experiment path builds that dict itself, so anything an evaluator
        # needs has to live where both paths can put it.
        turn_index = _nested(agent, "travel_agent", "turn_index")
        records.append({
            "span_id": span_id,
            "trace_id": agent["context.trace_id"],
            "session_id": _attr(agent, "attributes.session.id"),
            "prompt_version": _nested(agent, "travel_agent", "prompt_version"),
            "input": {
                "message": request,
                "turn_index": int(turn_index) if turn_index is not None else None,
            },
            "output": {
                "reply": reply,
                "tool_calls": tool_calls,
            },
        })
    if skipped:
        print(f"  skipped {len(skipped)} span(s) with no recorded input/output")
    _stamp_session_disclosure(records)
    return records


def _stamp_session_disclosure(records: list[dict]) -> None:
    """Record, per turn, the earliest turn in its session that disclosed.

    Some properties are true of a conversation rather than a turn. "The agent
    told this user it cannot book" is one: said at turn 1, it still holds at
    turn 4, and an evaluator that only sees turn 4 would score a correct
    conversation as a failure.

    This resolves it here rather than in the evaluator, because it is derived
    from the replies themselves — what the agent actually said, across the
    session — and not from the guardrail reporting on its own work. A monitor
    that reads the control's self-report cannot catch the control failing.
    """
    from common.capabilities import discloses

    earliest: dict[Any, int] = {}
    for rec in records:
        session, turn = rec.get("session_id"), rec["input"].get("turn_index")
        if session is None or turn is None:
            continue
        if discloses(rec["output"].get("reply") or ""):
            earliest[session] = min(earliest.get(session, turn), turn)

    for rec in records:
        rec["input"]["disclosed_by_turn"] = earliest.get(rec.get("session_id"))


def score(
    records: Iterable[dict],
    agent: str,
    *,
    judge_max_turns: int | None = None,
    rng: random.Random | None = None,
    registry: Any = None,
) -> pd.DataFrame:
    """Run every online-capable evaluator over the reconstructed records.

    `judge_max_turns` caps how many turns the **LLM** evaluators see; the code
    evaluators always run on everything. That asymmetry is the point. Code
    evaluators are free, and the ones that matter most here are invariants —
    `no_unredacted_pii` on a sampled 20% of traffic is not a privacy control.
    Judges cost a model call per turn per judge, and they are *signals*: a rate
    estimated on a random 150 turns is nearly as good as one over 3,000, at 5%
    of the bill. At the customer's ~114 conversations/hour an uncapped sweep is
    roughly 7,000 judge calls a day.

    It is a ceiling, not a rate — below it nothing is sampled, so thin traffic
    is never thinned further into the `min_sample` floor. The sample is drawn
    once and shared by every judge, so their rates stay comparable to each
    other; sampling per judge would have them grading different windows.

    `registry` defaults to the global one; tests pass their own so that
    exercising the sampling logic does not bill a real judge call per turn.
    """
    records = list(records)
    judge_records = records
    if judge_max_turns is not None and len(records) > judge_max_turns:
        judge_records = (rng or random.Random()).sample(records, judge_max_turns)

    rows = []
    for reg in (registry or REGISTRY).for_agent(agent, online=True):
        for rec in records if reg.kind == "code" else judge_records:
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
    judge_max_turns: int | None = None,
    dry_run: bool = False,
    client: Client | None = None,
) -> pd.DataFrame:
    """Sample recent spans, evaluate them, and annotate them in place.

    `limit` counts **spans**, not turns — one turn is ~8 spans here (the agent
    span, its LLM calls, and a span per tool call). Sizing it as if it were a
    turn cap silently truncates the window: a sweep configured at 200 scored 26
    turns and skipped every judge for falling under `min_sample`, which read as
    thin traffic and was really the fetch cap.
    """
    client = client or Client()
    start = datetime.now(UTC) - timedelta(minutes=since_minutes)
    spans = client.spans.get_spans_dataframe(
        project_identifier=project, start_time=start, limit=limit
    )
    records = reconstruct(spans, agent_span_name=agent_span_name)
    if not records:
        return pd.DataFrame()

    results = score(records, agent, judge_max_turns=judge_max_turns)
    if not dry_run and not results.empty:
        client.spans.log_span_annotations_dataframe(
            dataframe=results.dropna(subset=["score"]).set_index("span_id"),
            sync=True,
        )
    return results
