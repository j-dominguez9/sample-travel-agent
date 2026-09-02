"""Turn flagged production turns into candidate golden cases.

This is the task that makes the DAG a feedback loop rather than monitoring. The
golden dataset can only contain failures somebody thought to write down; the
duration-and-"direct flight" hallucination was 36% of live traffic and had zero
coverage, because nobody knew it existed until the monitor found it. Curation is
how that class stops being invisible on the next run.

Two properties matter:

**Candidates are never merged automatically.** They land in a separate Phoenix
dataset for review. The customer was explicit that regressions need a human in
the loop, and a loop that writes its own test cases from its own failures can
just as easily enshrine a wrong answer as a right one.

**Expected values are derived where they are derivable.** For a tool-backed
query the fixtures still say what the answer should have been, so a candidate
arrives fully labelled. For a reply-level failure — an unsourced claim — there
is nothing to compute, and the case is marked for review instead of guessed at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from evals.core.config import AgentConfig
from evals.core.registry import REGISTRY


@dataclass(frozen=True)
class Candidate:
    span_id: str
    message: str
    flagged_by: list[str]
    expected: dict[str, Any]
    needs_review: bool
    reason: str


def _is_iso_date(value: object) -> bool:
    try:
        date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        return False
    return True


def _failing(results: pd.DataFrame, agent: str) -> dict[str, list[str]]:
    """span_id -> evaluators that judged it a failure, direction-aware.

    A quiet window is the normal case, not an error: an hourly sweep over a
    period with no traffic returns an empty frame with no columns at all, and
    indexing it would fail the task. Monitoring that breaks when nothing
    happened is worse than no monitoring, because the failure looks like an
    incident.
    """
    directions = {r.name: r.direction for r in REGISTRY.for_agent(agent)}
    out: dict[str, list[str]] = {}
    if results.empty or "score" not in results.columns:
        return out
    for _, row in results.dropna(subset=["score"]).iterrows():
        name = row["annotation_name"]
        # Span annotations include human ones (`user_feedback`) and anything the
        # UI wrote. Only this agent's registered evaluators are graded here: an
        # unknown name would fall back to "maximize", turning a human
        # thumbs-down into a curation candidate attributed to an evaluator.
        if name not in directions:
            continue
        bad = (
            row["score"] >= 1.0
            if directions[name] == "minimize"
            else row["score"] <= 0.0
        )
        if bad:
            out.setdefault(row["span_id"], []).append(name)
    return out


def derive_expected(record: dict, truth_module: Any) -> tuple[dict[str, Any], bool, str]:
    """Compute what the tools should have returned, where the fixtures can say.

    Returns (expected, needs_review, reason).
    """
    calls = (record.get("output") or {}).get("tool_calls") or []
    if not calls:
        return ({}, True, "no tool call — expected behaviour needs a human")

    call = calls[0]
    args = call.get("input") or {}
    name = call.get("name")

    if name == "search_flights" and {"origin", "destination", "date"} <= set(args):
        # `expected_flights` compares ISO strings lexicographically and never
        # raises, so a malformed date silently matches nothing. Deriving from
        # that would publish "there are no flights on this route" as ground
        # truth when the real defect was the date the agent sent.
        if not _is_iso_date(args["date"]):
            return ({}, True, f"agent sent a malformed date ({args['date']!r})")
        flights = truth_module.expected_flights(
            args["origin"], args["destination"], args["date"]
        )
        return (
            {
                "expected_tool": name,
                "expected_args": args,
                "expected_result_ids": sorted(f["flight_number"] for f in flights),
                "expected_behavior": "answer" if flights else "empty",
            },
            False,
            "derived from fixtures",
        )

    if name == "search_hotels" and {"city", "check_in"} <= set(args):
        if not _is_iso_date(args["check_in"]):
            return ({}, True, f"agent sent a malformed date ({args['check_in']!r})")
        hotels = truth_module.expected_hotels(args["city"], args["check_in"])
        return (
            {
                "expected_tool": name,
                "expected_args": args,
                "expected_result_ids": sorted(h["name"] for h in hotels),
                "expected_behavior": "answer" if hotels else "empty",
            },
            False,
            "derived from fixtures",
        )

    # Reply-level failures (an unsourced claim over correct tool output) have no
    # computable expected value. The tool call was right; the prose was not.
    return (
        {"expected_tool": name, "expected_args": args},
        True,
        "tool call correct — the failure is in the reply, so a human decides",
    )


def collect(
    records: list[dict],
    results: pd.DataFrame,
    config: AgentConfig,
    truth_module: Any,
) -> list[Candidate]:
    failing = _failing(results, config.agent)
    by_span = {r["span_id"]: r for r in records}
    limit = int(config.curation.get("max_per_run", 25))

    candidates: list[Candidate] = []
    for span_id, evaluators in sorted(failing.items()):
        record = by_span.get(span_id)
        if record is None:
            continue
        expected, needs_review, reason = derive_expected(record, truth_module)
        candidates.append(
            Candidate(
                span_id=span_id,
                message=(record.get("input") or {}).get("message", ""),
                flagged_by=sorted(evaluators),
                expected=expected,
                needs_review=needs_review,
                reason=reason,
            )
        )
        if len(candidates) >= limit:
            break
    return candidates


def publish(candidates: list[Candidate], config: AgentConfig, client: Any = None) -> str | None:
    """Upsert candidates into their own Phoenix dataset for review."""
    if not candidates:
        return None
    from phoenix.client import Client

    client = client or Client()
    name = config.curation.get("candidates_dataset", f"{config.agent}-candidates")
    stamp = datetime.now(UTC).isoformat(timespec="seconds")

    dataset = client.datasets.create_dataset(
        name=name,
        examples=[
            {
                # Stable on the span, so a re-run updates rather than duplicates.
                "id": f"cand-{c.span_id}",
                "input": {"message": c.message},
                "output": c.expected,
                "metadata": {
                    "source": "production",
                    "span_id": c.span_id,
                    "flagged_by": c.flagged_by,
                    "needs_review": c.needs_review,
                    "reason": c.reason,
                    "collected_at": stamp,
                },
            }
            for c in candidates
        ],
    )
    return dataset.id


def summary(candidates: list[Candidate]) -> str:
    if not candidates:
        return "no new candidates"
    auto = sum(1 for c in candidates if not c.needs_review)
    lines = [
        (
            f"{len(candidates)} candidate case(s): {auto} fully labelled from "
            f"fixtures, {len(candidates) - auto} need review"
        )
    ]
    for c in candidates[:8]:
        mark = "review" if c.needs_review else "auto  "
        lines.append(f"  [{mark}] {','.join(c.flagged_by):40} {c.message[:52]}")
    return "\n".join(lines)


if __name__ == "__main__":  # manual smoke against the live project
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evals.agents.travel import evaluators, judges, truth  # noqa: F401
    from evals.core import config as cfg
    from evals.core import monitor

    conf = cfg.load(Path(__file__).resolve().parents[1] / "agents/travel/config.yaml")
    spans = monitor.Client().spans.get_spans_dataframe(
        project_identifier=conf.project, limit=300
    )
    recs = monitor.reconstruct(spans, agent_span_name=conf.agent_span_name)
    res = monitor.score(recs, conf.agent)
    print(summary(collect(recs, res, conf, truth)))
    print(json.dumps({"records": len(recs), "annotations": len(res)}))
