"""Turn a set of failing turns into something the person paged can act on.

An alert that says "hallucination is at 27%" tells the recipient a number. What
they need is which turns failed, what they have in common, what the evaluator
actually said, and what to try. The Head of Product is not going to open Phoenix
and read twenty-four traces at 3am.

The digest is built in two layers, and the separation is the point:

**Evidence — deterministic, always correct.** Which evaluators failed at what
rate, which turns, what those turns share, and the evaluators' own explanations
verbatim. This is aggregation, not inference. It is also what actually solves
most cases: the relative-date bug was obvious the moment four judge
explanations sat next to each other, all saying the same thing.

**Hypothesis — one LLM call, explicitly unverified.** Written *below* the
evidence so a reader can check it against the facts rather than take it on
trust, and it never replaces them.

One rule is baked into the hypothesis prompt: it must consider that the
*evaluator* may be wrong, not the agent. In this project's own history roughly
half the investigated failures were eval defects — a rubric that contradicted
its own examples, a validation set with mismatched pairs, judges scoring
degenerate inputs. A diagnosis that can only blame the agent would have been
wrong about half the time.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from evals.core.config import AgentConfig
from evals.core.registry import REGISTRY

#: Temporal expressions the agent has to resolve rather than hand back.
RELATIVE_DATE = re.compile(
    r"\b(next|this|last|coming|tomorrow|tonight|weekend|week|month|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


@dataclass
class Cluster:
    evaluator: str
    failed: int
    applicable: int
    threshold: float | None
    direction: str
    turns: list[dict] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        """The rate the threshold is expressed against, direction included.

        For a maximize evaluator that is the pass rate; for a minimize one it is
        the flagged rate, because `max_rate: 0.05` means "at most 5% flagged".
        Reporting a pass rate against a ceiling made every minimize evaluator
        with a single failure look permanently breached, and put a header on the
        digest that contradicted `thresholds.evaluate`.
        """
        if not self.applicable:
            return 0.0
        failed_rate = self.failed / self.applicable
        return failed_rate if self.direction == "minimize" else 1 - failed_rate

    def shared_features(self) -> list[str]:
        """Features true of every failing turn — the discriminating part."""
        if not self.turns:
            return []
        feats: list[Counter] = []
        for t in self.turns:
            calls = (t.get("output") or {}).get("tool_calls") or []
            msg = (t.get("input") or {}).get("message", "")
            f = Counter()
            f["no tool call was made"] = int(not calls)
            f["every tool returned no results"] = int(
                bool(calls) and all(not c.get("output") for c in calls)
            )
            f["the request contains a relative date"] = int(bool(RELATIVE_DATE.search(msg)))
            for c in calls:
                f[f"called {c.get('name')}"] = 1
            feats.append(f)
        keys = set().union(*(f.keys() for f in feats))
        return sorted(k for k in keys if all(f.get(k) for f in feats))


def build(
    records: list[dict],
    results: pd.DataFrame,
    config: AgentConfig,
    *,
    breached_only: bool = False,
) -> list[Cluster]:
    """Group failing turns by evaluator.

    Deliberately not gated on a threshold breach by default: the relative-date
    failures that mattered most were suppressed by the thin-sample guard at
    n=9, and a breach-only analysis would have printed nothing about them.
    """
    if results.empty or "score" not in results.columns:
        return []

    directions = {r.name: r.direction for r in REGISTRY.for_agent(config.agent)}
    by_span = {r["span_id"]: r for r in records}
    scored = results.dropna(subset=["score"])

    clusters: list[Cluster] = []
    for name, rows in scored.groupby("annotation_name"):
        minimize = directions.get(str(name), "maximize") == "minimize"
        bad = rows[rows["score"] >= 1.0] if minimize else rows[rows["score"] <= 0.0]
        if bad.empty:
            continue

        limits = config.thresholds.get(str(name), {})
        limit = limits.get("max_rate") if minimize else limits.get("min_rate")
        cluster = Cluster(
            evaluator=str(name),
            failed=len(bad),
            applicable=len(rows),
            threshold=float(limit) if limit is not None else None,
            direction="minimize" if minimize else "maximize",
        )
        for _, row in bad.iterrows():
            turn = by_span.get(row["span_id"])
            if turn:
                cluster.turns.append(turn)
            if isinstance(row.get("explanation"), str) and row["explanation"]:
                cluster.explanations.append(row["explanation"])
        clusters.append(cluster)

    if breached_only:
        clusters = [c for c in clusters if c.threshold is not None and (
            c.rate < c.threshold if c.direction == "maximize" else c.rate > c.threshold
        )]
    return sorted(clusters, key=lambda c: -c.failed)


def evidence(clusters: list[Cluster], limit_turns: int = 4) -> str:
    """The deterministic half. Facts only — no inference."""
    if not clusters:
        return "No failing turns in this window."
    out: list[str] = []
    for c in clusters:
        head = f"{c.evaluator}: {c.failed} of {c.applicable} applicable turns failed"
        if c.threshold is not None:
            label = "flagged" if c.direction == "minimize" else "pass"
            comparison = "max" if c.direction == "minimize" else "min"
            head += f"  ({label} rate {c.rate:.0%}, {comparison} {c.threshold:.0%})"
        out.append(head)

        shared = c.shared_features()
        if shared:
            out.append("  shared by every failing turn:")
            out += [f"    - {s}" for s in shared]

        out.append("  failing turns:")
        for t in c.turns[:limit_turns]:
            msg = " ".join((t.get("input") or {}).get("message", "").split())
            reply = " ".join(((t.get("output") or {}).get("reply") or "").split())
            out.append(f'    Q: {msg[:100]}')
            out.append(f'    A: {reply[:120]}')
        if len(c.turns) > limit_turns:
            out.append(f"    ... and {len(c.turns) - limit_turns} more")

        if c.explanations:
            out.append("  what the evaluator said:")
            for e in c.explanations[:limit_turns]:
                out.append(f'    "{" ".join(e.split())[:180]}"')
        out.append("")
    return "\n".join(out)


HYPOTHESIS_PROMPT = """You are helping an on-call engineer triage failing evaluations of an AI travel agent.

Below is evidence from one monitoring window: which evaluators failed, what the
failing turns have in common, the agent's replies, and the evaluators' own
explanations.

<evidence>
{evidence}
</evidence>

The agent has four tools — search_flights, search_hotels, get_weather,
create_itinerary — backed by fixed local data. Its system prompt tells it to
give concrete options, avoid clarifying questions, and never mention internal
systems or data sources.

Consider all three possibilities before concluding, and say which you believe:
  AGENT      the agent genuinely behaved wrongly
  EVALUATOR  the agent behaved acceptably and the evaluator or its rubric is
             wrong for these cases (mis-scoped, contradictory criteria, or
             scoring inputs it cannot meaningfully judge)
  DATA       the fixtures or the request are at fault, not the behaviour

Answer in this exact form, under 140 words total:

CAUSE: <one sentence naming the single most likely root cause>
CATEGORY: <AGENT | EVALUATOR | DATA>
CONFIDENCE: <high | medium | low>
EVIDENCE FOR: <the specific detail above that supports this>
EVIDENCE AGAINST: <what would contradict it, or 'none noted'>
PROPOSED: <one concrete change, naming the file or prompt section to edit>
CHECK FIRST: <the cheapest thing to run to confirm before changing anything>"""


def hypothesis(clusters: list[Cluster], *, model: str | None = None) -> str:
    """One LLM call over the evidence. Explicitly a hypothesis, never a verdict."""
    if not clusters:
        return ""
    import anthropic

    model = model or os.getenv("DIAGNOSE_MODEL", "claude-sonnet-4-6")
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": HYPOTHESIS_PROMPT.format(evidence=evidence(clusters, limit_turns=6)),
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def from_experiment(client: Any, dataset_id: str, experiment_name: str) -> tuple[list[dict], pd.DataFrame]:
    """Adapt a Phoenix experiment into the (records, scores) shape `digest` wants.

    The monitoring path reads spans; the regression path reads experiment runs.
    Both end up as the same `{reply, tool_calls}` record plus a scores frame, so
    one diagnosis routine serves both rather than two that drift apart.
    """
    from evals.report import _g, load, resolve

    scores_by_run = load(client, dataset_id, experiment_name)[0]
    experiment = client.experiments.get_experiment(
        experiment_id=resolve(client, dataset_id, experiment_name)
    )

    records: list[dict] = []
    for run in experiment["task_runs"]:
        output = _g(run, "output") or {}
        if not isinstance(output, dict):
            continue
        records.append({
            # `digest` keys turns by span_id; the run id plays that role here.
            "span_id": str(_g(run, "id")),
            "input": {"message": ""},
            "output": {
                "reply": output.get("reply", ""),
                "tool_calls": output.get("tool_calls") or [],
            },
        })

    rows = [
        {"span_id": str(run_id), "annotation_name": name, "score": value,
         "label": None, "explanation": None}
        for (run_id, name), value in scores_by_run.items()
    ]
    return records, pd.DataFrame(rows)


def digest(
    records: list[dict],
    results: pd.DataFrame,
    config: AgentConfig,
    *,
    with_hypothesis: bool = True,
) -> tuple[str, list[Cluster]]:
    clusters = build(records, results, config)
    if not clusters:
        return ("No failing turns in this window.", [])

    parts = ["EVIDENCE (deterministic)", "=" * 60, evidence(clusters)]
    if with_hypothesis:
        try:
            parts += [
                "HYPOTHESIS (generated, unverified — check it against the evidence above)",
                "=" * 60,
                hypothesis(clusters),
            ]
        except Exception as exc:  # noqa: BLE001
            # The evidence is the load-bearing half. If the hypothesis call
            # fails, the alert must still carry the facts rather than nothing.
            parts += [f"(hypothesis unavailable: {type(exc).__name__}: {exc})"]
    return ("\n".join(parts), clusters)
