"""Compare two experiments and print a before/after scorecard.

    python -m evals.report baseline-main current-fixed

Two things this report is careful about, because both would otherwise flatter
the results:

1. **Applicability.** A code evaluator returns True on examples it does not
   apply to — `itinerary_day_count` passes trivially on a hotel query. Scoring
   over all 199 examples therefore dilutes every evaluator toward 100% and
   makes a real failure look small. Rates here are computed only over the
   examples an evaluator actually governs, and `n` says how many that is.

2. **Defect class.** A cell where `main` had the code and the data but gave the
   wrong answer is a fixed bug. A cell where the code path and the fixture
   fields did not exist is an added capability. Reporting them as one number
   would overstate the regression story.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _g(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


# --------------------------------------------------------------------------
# Which examples each evaluator actually governs
# --------------------------------------------------------------------------

APPLIES = {
    "tool_selection":           lambda m: True,
    "date_grounding":           lambda m: bool(m["args"].get("date")),
    "tool_result_exact":        lambda m: m["tool"] in {"search_flights", "search_hotels"},
    "flight_direction":         lambda m: m["tool"] == "search_flights",
    "no_fabricated_flights":    lambda m: m["tool"] == "search_flights",
    "itinerary_day_count":      lambda m: m["tool"] == "create_itinerary",
    "weather_values_plausible": lambda m: m["tool"] == "get_weather",
    "no_internal_leak":         lambda m: m["behavior"] in {"empty", "out_of_scope"},
    "stays_in_scope":           lambda m: m["behavior"] == "out_of_scope",
    # Both grounding judges compare the reply against the tool results. On an
    # out-of-scope turn no tool runs, so their source of truth is empty and
    # every substantive sentence scores as unsupported — a degenerate input,
    # not a finding. Scope adherence there is covered by `stays_in_scope` and
    # `graceful_alternative`, which are built for it.
    "hallucination":            lambda m: m["behavior"] != "out_of_scope",
    "recommendation_grounded":  lambda m: m["behavior"] != "out_of_scope",
}

# Defect class per dimension cell. See module docstring.
CATEGORY = {
    "route:direction_pair": "bugfix",       # set-comparison ignored direction
    "route:reverse_only":   "bugfix",
    "route:forward":        "bugfix",       # bidirectional routes leak the reverse leg
    "tools:flight+weather": "bugfix",
    "tools:flight+hotel":   "bugfix",
    "date_window:full":     "capability",   # date filtering: new code AND new fields
    "date_window:partial":  "capability",
    "date:window_closed":   "unchanged",    # hotels already filtered on check_in
    "route:absent":         "unchanged",
    "city:no_data":         "unchanged",
    "scope:out":            "unchanged",
    "date:relative":        "unchanged",
}

# A few evaluators are defect-specific regardless of which cell they land in.
EVALUATOR_CATEGORY = {"weather_values_plausible": "bugfix"}  # the C-to-F bug


def category(cell: str, evaluator: str) -> str:
    if evaluator in EVALUATOR_CATEGORY:
        return EVALUATOR_CATEGORY[evaluator]
    if cell.startswith("num_days:"):
        return "bugfix"                      # itinerary off-by-one
    return CATEGORY.get(cell, "unchanged")


def resolve(client, dataset_id: str, name: str) -> str:
    matches = [e for e in client.experiments.list(dataset_id=dataset_id)
               if _g(e, "name") == name]
    if not matches:
        avail = sorted(str(_g(e, "name")) for e in client.experiments.list(dataset_id=dataset_id))
        raise SystemExit(f"no experiment named {name!r}. available: {avail}")
    return str(_g(matches[-1], "id"))


def load_by_id(client, experiment_id: str):
    """Return (scores, run->example) for one experiment, addressed by id.

    Addressing by id matters wherever two experiments can share a name — a task
    retry or a same-day re-trigger produces exactly that, and resolving by name
    would silently pick the wrong one.

    Scores are keyed per *run*, not per example: keying by example would
    collapse `repetitions=N` to a single observation and silently discard the
    rest, which is the noise the repetitions exist to average out.
    """
    exp = client.experiments.get_experiment(experiment_id=experiment_id)
    run_example = {_g(r, "id"): _g(r, "dataset_example_id") for r in exp["task_runs"]}
    scores: dict[tuple[str, str], float] = {}
    for e in exp["evaluation_runs"]:
        res = _g(e, "result")
        if res is None:
            continue
        scores[(_g(e, "experiment_run_id"), str(_g(e, "name")))] = float(
            _g(res, "score") or 0.0
        )
    return scores, run_example


def load(client, dataset_id: str, name: str):
    """Same, addressed by experiment name."""
    return load_by_id(client, resolve(client, dataset_id, name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--dataset", default="travel-golden-v1")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    from phoenix.client import Client

    client = Client()
    ds = client.datasets.get_dataset(dataset=args.dataset)
    before, before_runs = load(client, ds.id, args.before)
    after, after_runs = load(client, ds.id, args.after)
    run_example = {**before_runs, **after_runs}

    # Runs reference the base64 node_id, not the stable example id we assigned.
    meta: dict[str, dict] = {}
    for ex in ds.examples:
        out = ex.get("output") or {}
        meta[ex.get("node_id")] = {
            **(ex.get("metadata") or {}),
            "example": ex.get("id"),
            "tool": out.get("expected_tool"),
            "args": out.get("expected_args") or {},
            "behavior": out.get("expected_behavior"),
        }

    evaluators = sorted({k[1] for k in before} | {k[1] for k in after})

    def _meta(run_id):
        return meta.get(run_example.get(run_id))

    def rate(scores, ev, extra=lambda m: True):
        applies = APPLIES.get(ev, lambda m: True)
        vals = [
            v for (rid, e), v in scores.items()
            if e == ev and (m := _meta(rid)) is not None and applies(m) and extra(m)
        ]
        return (sum(vals) / len(vals) * 100 if vals else float("nan")), len(vals)

    print(f"\n{'evaluator':26} {'before':>9} {'after':>9} {'delta':>10}  applicable")
    print("-" * 70)
    for ev in evaluators:
        b, nb = rate(before, ev)
        a, na = rate(after, ev)
        if math.isnan(b):        # no applicable examples for this evaluator
            continue
        d = a - b
        arrow = " " if abs(d) < 0.05 else ("^" if d > 0 else "v")
        print(f"{ev:26} {b:8.1f}% {a:8.1f}%  {arrow}{d:+7.1f}   {max(nb, na):>4}")

    print("\n" + "=" * 70)
    print("SPLIT BY DEFECT CLASS")
    print("=" * 70)
    for cat, blurb in (
        ("bugfix", "same fixtures, same code path — main gave a wrong answer"),
        ("capability", "date-aware availability — new code AND new fixture fields"),
        ("unchanged", "behaviour identical in both — control group"),
    ):
        print(f"\n  {cat.upper()}  ({blurb})")
        any_row = False
        for ev in evaluators:
            pred = lambda m, e=ev, c=cat: category(m.get("cell", "?"), e) == c
            b, nb = rate(before, ev, pred)
            a, na = rate(after, ev, pred)
            if math.isnan(b) or max(nb, na) == 0:
                continue
            any_row = True
            print(f"      {ev:26} {b:6.1f}%  ->  {a:6.1f}%   ({a - b:+6.1f})   n={max(nb, na)}")
        if not any_row:
            print("      (no applicable evaluators)")

    # Per-cell movement, applicability-aware.
    cb: dict[str, list[float]] = defaultdict(list)
    ca: dict[str, list[float]] = defaultdict(list)
    for scores, bucket in ((before, cb), (after, ca)):
        for (rid, ev), v in scores.items():
            m = _meta(rid)
            if m and APPLIES.get(ev, lambda _: True)(m):
                bucket[m.get("cell", "?")].append(v)

    rows = []
    for cell in sorted(set(cb) | set(ca)):
        b = sum(cb[cell]) / len(cb[cell]) * 100 if cb[cell] else float("nan")
        a = sum(ca[cell]) / len(ca[cell]) * 100 if ca[cell] else float("nan")
        rows.append((a - b, cell, b, a, len(ca[cell] or cb[cell])))
    rows.sort(reverse=True)

    print(f"\n{'dimension cell':26} {'before':>9} {'after':>9} {'delta':>10}    n")
    print("-" * 70)
    for d, cell, b, a, n in rows:
        print(f"{cell:26} {b:8.1f}% {a:8.1f}%  {d:+8.1f}   {n:>4}")


if __name__ == "__main__":
    main()
