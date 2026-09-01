"""Turn a sweep's scores into breaches worth waking someone for.

Three rules keep this from becoming an alarm that gets muted:

1. **Direction comes from the registry, not the config.** `hallucination` scores
   1.0 for the bad outcome; everything else scores 1.0 for the good one. A
   config that had to restate that would eventually restate it wrongly, and the
   dashboard would read backwards.
2. **A thin sample is not a signal.** An early sweep put `graceful_alternative`
   at 50% on n=2. Alerting on that is alerting on one conversation.
3. **Paging is narrower than alerting.** Everything with a threshold is
   reported; only the classes the customer named as page-worthy escalate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from evals.core.config import AgentConfig
from evals.core.registry import REGISTRY


@dataclass(frozen=True)
class Breach:
    evaluator: str
    rate: float
    limit: float
    direction: str
    n: int
    page: bool

    def describe(self) -> str:
        comparison = "above" if self.direction == "minimize" else "below"
        return (
            f"{self.evaluator}: {self.rate:.1%} is {comparison} the "
            f"{self.limit:.0%} limit over {self.n} turns"
        )


@dataclass(frozen=True)
class Report:
    agent: str
    breaches: list[Breach]
    measured: dict[str, tuple[float, int]]
    skipped: dict[str, int]
    #: Configured evaluators that produced no usable score at all — a judge
    #: outage, a bad key, a rate limit. Silence here is not health.
    unscored: list[str] = field(default_factory=list)

    @property
    def should_page(self) -> bool:
        return any(b.page for b in self.breaches)

    @property
    def is_degraded(self) -> bool:
        """True when a configured evaluator produced nothing to judge."""
        return bool(self.unscored)

    def summary(self) -> str:
        lines = [f"[{self.agent}] {len(self.breaches)} threshold breach(es)"]
        for name, (rate, n) in sorted(self.measured.items()):
            flag = "  BREACH" if any(b.evaluator == name for b in self.breaches) else ""
            lines.append(f"  {name:26} {rate:6.1%}  n={n:<4}{flag}")
        for name, n in sorted(self.skipped.items()):
            lines.append(f"  {name:26} skipped — n={n} below minimum")
        for name in sorted(self.unscored):
            lines.append(f"  {name:26} NO USABLE SCORES — evaluator may be failing")
        return "\n".join(lines)


def evaluate(results: pd.DataFrame, config: AgentConfig) -> Report:
    """Compare a sweep's annotation scores against the agent's thresholds."""
    directions = {r.name: r.direction for r in REGISTRY.for_agent(config.agent)}
    page_on = set(config.alerting.get("page_on") or [])

    breaches: list[Breach] = []
    measured: dict[str, tuple[float, int]] = {}
    skipped: dict[str, int] = {}
    unscored: list[str] = []

    if results.empty or "score" not in results.columns:
        return Report(config.agent, breaches, measured, skipped, unscored)

    scored = results.dropna(subset=["score"])
    for name, limits in config.thresholds.items():
        rows = scored[scored["annotation_name"] == name]
        if rows.empty:
            # Distinguish "this evaluator was never applicable" from "it ran and
            # every call errored". `monitor.score` records a failed judge as
            # score=None, so a total outage leaves rows here but none scored —
            # and reporting nothing would render an outage as a clean bill.
            attempted = results[results["annotation_name"] == name]
            if not attempted.empty:
                unscored.append(str(name))
            continue
        n = len(rows)
        rate = float(rows["score"].mean())

        if n < config.min_sample:
            skipped[name] = n
            continue
        measured[name] = (rate, n)

        direction = directions.get(name, "maximize")
        if direction == "minimize":
            limit = float(limits.get("max_rate", 1.0))
            breached = rate > limit
        else:
            limit = float(limits.get("min_rate", 0.0))
            breached = rate < limit

        if breached:
            breaches.append(
                Breach(name, rate, limit, direction, n, page=name in page_on)
            )

    return Report(config.agent, breaches, measured, skipped, unscored)
