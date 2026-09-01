"""Tier B — agent integration evals. Calls the model, so it gates PRs, not pushes.

The split that shapes this file is invariants versus signals:

  invariant   exactly one acceptable behaviour, checkable in code.
              `assert` it — a failure turns CI red like any unit test.
  signal      lives on a spectrum with no single correct string.
              `log_evaluation` only — never assert. Judge output is
              non-deterministic, and we have direct evidence: the leak metric
              moved 6.7 points on a single run and turned out to be sampling
              noise. A judge in a blocking gate flakes builds, and a flaky gate
              gets disabled.

Signals still gate — on the aggregate, nightly, not per case. That job belongs
to `evals/run_experiment.py`, not here.

Cases are a representative subset of the golden dataset rather than all 199: a
PR gate should be fast and cover every dimension cell, and the full suite runs
on a schedule. Stable `ids` keep each case the same Phoenix example across runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from phoenix.client.pytest import log_evaluation, log_output

from agent.loop import run_agent
from evals.agents.travel import truth as T
from evals.core.runner import extract_tool_calls

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "data" / "travel_golden_v1.json"


def _subset() -> list[dict]:
    """Up to two examples per dimension cell — full coverage, ~a fifth the cost."""
    rows = json.loads(GOLDEN.read_text())
    per_cell: dict[str, list[dict]] = {}
    for row in rows:
        per_cell.setdefault(row["metadata"]["cell"], []).append(row)
    picked = [r for cell in sorted(per_cell) for r in per_cell[cell][:2]]
    return sorted(picked, key=lambda r: r["id"])


CASES = _subset()
IDS = [c["id"] for c in CASES]


def _run(message: str) -> dict:
    reply, history = run_agent([{"role": "user", "content": message}])
    return {"reply": reply, "tool_calls": extract_tool_calls(history)}


@pytest.mark.phoenix(dataset="travel-agent-ci")
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_agent_turn(case: dict) -> None:
    expected = case["output"]
    result = _run(case["input"]["message"])
    log_output(result)

    calls = result["tool_calls"]
    called = {c["name"] for c in calls}

    # Whether the model *chooses* to call a tool is stochastic. Measured over
    # 597 runs, tool_selection sits at 98.8-99.2%, so asserting it per case
    # across 38 cases turns ~a third of CI runs red for no reason — and a gate
    # that flakes gets switched off. It is logged here and gated on the
    # aggregate by the nightly experiment instead.
    #
    # Everything below asserts on the *content* of calls that did happen, which
    # is deterministic given the call: the fixtures decide it, not the model.
    want_tool = expected["expected_tool"]
    if want_tool is None:
        selected = not called
        detail = f"out-of-scope query reached {called}"
    else:
        selected = want_tool in called
        detail = f"expected {want_tool}, agent called {called or 'nothing'}"
    log_evaluation(
        name="tool_selection",
        score=1.0 if selected else 0.0,
        explanation=None if selected else detail,
    )
    if want_tool is not None and not selected:
        pytest.skip(f"no {want_tool} call this run — nothing to check the content of")

    checks: list[tuple[str, bool, str]] = []

    want_date = (expected.get("expected_args") or {}).get("date")
    if want_date:
        dates = {(c.get("input") or {}).get("date") for c in calls if c["name"] == want_tool}
        checks.append((
            "date_grounding",
            want_date in dates,
            f"date reached the tool as {dates}, expected {want_date}",
        ))

    for call in calls:
        if call["name"] != "search_flights" or not isinstance(call.get("output"), list):
            continue
        args = call.get("input") or {}
        legal = {f["flight_number"] for f in T._legs(args.get("origin", ""), args.get("destination", ""))}
        returned = {r.get("flight_number") for r in call["output"] if isinstance(r, dict)}
        checks.append((
            "flight_direction",
            returned <= legal,
            f"wrong-direction flights: {sorted(returned - legal)}",
        ))

    if want_tool in {"search_flights", "search_hotels"}:
        key = "flight_number" if want_tool == "search_flights" else "name"
        for call in calls:
            if call["name"] == want_tool and isinstance(call.get("output"), list):
                got = {str(r.get(key)) for r in call["output"] if isinstance(r, dict)}
                checks.append((
                    "tool_result_exact",
                    got == set(expected["expected_result_ids"]),
                    (
                        f"tool returned {sorted(got)}, fixtures say "
                        f"{sorted(expected['expected_result_ids'])}"
                    ),
                ))
                break

    # Regression cases target ~100% and gate the build. Capability cases target
    # 50-80% by design — asserting on them would make a red build the expected
    # state, so they run here for coverage and trend in Phoenix instead.
    gates = case["metadata"]["suite"] == "regression"
    for name, ok, detail in checks:
        log_evaluation(name=name, score=1.0 if ok else 0.0, explanation=None if ok else detail)
        if gates:
            assert ok, detail

    # ---- signals: logged and trended, never asserted --------------------
    log_evaluation(
        name="tool_call_count",
        score=float(len(calls)),
        metadata={"cell": case["metadata"]["cell"]},
    )
    if expected["expected_behavior"] in {"empty", "out_of_scope"}:
        from evals.agents.travel.evaluators import LEAK_PHRASES

        leaked = [p for p in LEAK_PHRASES if p in result["reply"].lower()]
        log_evaluation(
            name="no_internal_leak",
            score=0.0 if leaked else 1.0,
            label="leaked" if leaked else "clean",
            explanation=f"matched {leaked}" if leaked else None,
        )
