"""Cost and token reporting, read from Phoenix rather than recomputed.

The discovery call asked for "a separate dashboard for cost and token usage",
and Phoenix already answers most of it: it ships a pricing table, matches spans
to it by model name, and exposes per-project cost through GraphQL. Its own UI is
the exploratory view — this module exists for the two things that view cannot
do on its own.

**Per conversation, not per project.** The customer thinks in cost per
conversation; Phoenix reports totals. Dividing by sessions is what turns a
number nobody can act on into one that can be compared against a unit economic.

**Agent cost and eval cost side by side.** Running this system is itself an
operating expense, and on current traffic the evaluation stack costs *more than
the agent it watches*. A cost dashboard that reports only the agent hides the
part of the bill the customer is choosing to take on by adopting this, which is
exactly the number they need before scaling it to 1M conversations a year.

Pricing deliberately stays in Phoenix. Restating rates here would create a
second source of truth that drifts the moment a provider changes a price, and
Phoenix already tracks per-model prices including cache reads and writes.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

_QUERY = """
query($tr: TimeRange!) {
  projects(first: 200) {
    edges {
      node {
        name
        traceCount(timeRange: $tr)
        sessionCount(timeRange: $tr)
        costSummary(timeRange: $tr) {
          total { tokens cost }
          prompt { tokens cost }
          completion { tokens cost }
        }
      }
    }
  }
}
"""


def _endpoint() -> str:
    ep = os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or os.getenv("PHOENIX_ENDPOINT")
    if not ep:
        raise KeyError("PHOENIX_COLLECTOR_ENDPOINT is not set")
    return ep.rstrip("/")


def fetch_projects(*, since_minutes: int, endpoint: str | None = None) -> dict[str, dict]:
    """Raw per-project cost for a window, keyed by exact project name.

    Phoenix's project filter matches on substring, so `travel-agent` also
    returns `travel-agent-evals`. Every project is fetched and matched exactly
    here instead — silently summing a neighbouring project into the agent's
    bill is the kind of error a dashboard never surfaces.
    """
    end = datetime.now(UTC)
    payload = json.dumps({
        "query": _QUERY,
        "variables": {
            "tr": {"start": (end - timedelta(minutes=since_minutes)).isoformat(),
                   "end": end.isoformat()},
        },
    }).encode()
    request = urllib.request.Request(
        f"{endpoint or _endpoint()}/graphql", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.load(response)
    if body.get("errors"):
        raise RuntimeError(f"Phoenix cost query failed: {body['errors']}")
    return {
        edge["node"]["name"]: edge["node"]
        for edge in body["data"]["projects"]["edges"]
    }


def _totals(node: dict | None) -> tuple[float, float]:
    """(cost, tokens) for a project node, tolerating an untraced window."""
    if not node:
        return 0.0, 0.0
    total = ((node.get("costSummary") or {}).get("total")) or {}
    return float(total.get("cost") or 0.0), float(total.get("tokens") or 0.0)


@dataclass(frozen=True)
class CostReport:
    agent: str
    window_minutes: int
    conversations: int
    turns: int
    agent_cost: float
    agent_tokens: float
    eval_cost: float
    eval_tokens: float
    annual_conversations: int | None = None
    max_cost_per_conversation: float | None = None
    missing_projects: list[str] = field(default_factory=list)

    @property
    def cost_per_conversation(self) -> float | None:
        return self.agent_cost / self.conversations if self.conversations else None

    @property
    def cost_per_turn(self) -> float | None:
        return self.agent_cost / self.turns if self.turns else None

    @property
    def tokens_per_turn(self) -> float | None:
        return self.agent_tokens / self.turns if self.turns else None

    @property
    def eval_overhead(self) -> float | None:
        """Eval spend as a fraction of agent spend. >1.0 means evals cost more."""
        return self.eval_cost / self.agent_cost if self.agent_cost else None

    @property
    def projected_annual_cost(self) -> float | None:
        """Agent spend at the customer's stated volume, at the current unit rate.

        A projection, not a forecast: it assumes today's mix of conversation
        types and today's prices. It is here because "$1.78 this window" is not
        a number anyone can plan against.
        """
        per = self.cost_per_conversation
        if per is None or self.annual_conversations is None:
            return None
        return per * self.annual_conversations

    @property
    def breaches(self) -> list[str]:
        ceiling, per = self.max_cost_per_conversation, self.cost_per_conversation
        if ceiling is None or per is None or per <= ceiling:
            return []
        return [f"cost per conversation ${per:.4f} is above the ${ceiling:.4f} ceiling"]

    def summary(self) -> str:
        lines = [f"[{self.agent}] cost over the last {self.window_minutes} min"]
        if not self.turns:
            lines.append("  no traffic in the window — nothing to report")
            return "\n".join(lines)

        def money(v: float | None, places: int = 4) -> str:
            return "n/a" if v is None else f"${v:,.{places}f}"

        lines += [
            f"  conversations           {self.conversations}",
            f"  turns                   {self.turns}",
            (f"  agent spend             {money(self.agent_cost)}"
             f"  ({self.agent_tokens:,.0f} tokens)"),
            f"  per conversation        {money(self.cost_per_conversation)}",
            f"  per turn                {money(self.cost_per_turn)}",
            f"  tokens per turn         {self.tokens_per_turn:,.0f}"
            if self.tokens_per_turn else "  tokens per turn         n/a",
            (f"  eval spend              {money(self.eval_cost)}"
             f"  ({self.eval_tokens:,.0f} tokens)"),
        ]
        if self.eval_overhead is not None:
            note = " — evals cost more than the agent" if self.eval_overhead > 1 else ""
            lines.append(f"  eval overhead           {self.eval_overhead:.0%} of agent spend{note}")
        if self.projected_annual_cost is not None:
            lines.append(
                f"  projected at {self.annual_conversations:,}/yr   "
                f"{money(self.projected_annual_cost, 2)}"
            )
        for breach in self.breaches:
            lines.append(f"  BREACH: {breach}")
        for name in self.missing_projects:
            # Named rather than silently zeroed: a cost report that quietly
            # drops a project under-reports the bill, which is the one failure
            # mode a cost dashboard must not have.
            lines.append(f"  note: project {name!r} has no traces in this window")
        return "\n".join(lines)


def report(config: Any, *, since_minutes: int | None = None,
           endpoint: str | None = None) -> CostReport:
    """Build a cost report for one agent from its config."""
    settings = getattr(config, "cost", None) or {}
    # `is None`, not truthiness: `--window 0` is a request for a zero-length
    # window, not an absent argument, and silently substituting the config
    # value would report a different window than the one asked for.
    window = (
        int(config.monitoring.get("window_minutes", 60))
        if since_minutes is None else since_minutes
    )
    projects = fetch_projects(since_minutes=window, endpoint=endpoint)

    agent_node = projects.get(config.project)
    eval_name = settings.get("eval_project")
    eval_node = projects.get(eval_name) if eval_name else None

    missing = [n for n in (config.project, eval_name) if n and n not in projects]
    agent_cost, agent_tokens = _totals(agent_node)
    eval_cost, eval_tokens = _totals(eval_node)

    return CostReport(
        agent=config.agent,
        window_minutes=window,
        conversations=int((agent_node or {}).get("sessionCount") or 0),
        turns=int((agent_node or {}).get("traceCount") or 0),
        agent_cost=agent_cost,
        agent_tokens=agent_tokens,
        eval_cost=eval_cost,
        eval_tokens=eval_tokens,
        annual_conversations=settings.get("annual_conversations"),
        max_cost_per_conversation=settings.get("max_cost_per_conversation"),
        missing_projects=missing,
    )


def main() -> None:
    """Print the cost report for each configured agent.

    Phoenix's own Dashboards page covers totals, trends and the per-model
    split, and is the right place to explore. What it cannot show is a rate:
    cost per conversation needs spend divided by sessions, and cost per turn
    needs it divided by traces. Those are the numbers the customer asked about,
    so they need somewhere to live — here, and in the monitoring DAG's
    `cost_report` task.
    """
    import argparse

    from evals.core import config as agent_config

    parser = argparse.ArgumentParser(description="Cost and token usage per agent.")
    parser.add_argument("--window", type=int, default=None, metavar="MINUTES",
                        help="look-back window (default: the agent's monitoring window)")
    args = parser.parse_args()

    for conf in agent_config.discover():
        print(report(conf, since_minutes=args.window).summary())
        print()


if __name__ == "__main__":
    main()
