"""Load per-agent configuration.

The DAGs discover agents by globbing `evals/agents/*/config.yaml`, so a new
tenant is a directory and a file. Nothing in the framework or the DAG changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

AGENTS_DIR = Path(__file__).resolve().parents[1] / "agents"


@dataclass(frozen=True)
class AgentConfig:
    agent: str
    project: str
    dataset: str
    agent_span_name: str
    monitoring: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    alerting: dict[str, Any] = field(default_factory=dict)
    regression: dict[str, Any] = field(default_factory=dict)
    curation: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def schedule(self) -> str:
        return self.monitoring.get("schedule", "@hourly")

    @property
    def min_sample(self) -> int:
        return int(self.monitoring.get("min_sample", 20))


def load(path: str | Path) -> AgentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    known = {f for f in AgentConfig.__dataclass_fields__ if f != "path"}
    # Dropping unknown keys silently is how `threshold:` for `thresholds:`
    # becomes an agent that monitors forever and never alerts. A config typo
    # should fail loudly at parse time, not degrade into a no-op.
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"{path}: unknown config key(s) {unknown}. Valid keys: {sorted(known)}"
        )
    return AgentConfig(**raw, path=path)


def discover() -> list[AgentConfig]:
    """Every configured agent, sorted for a stable DAG ordering."""
    return sorted(
        (load(p) for p in AGENTS_DIR.glob("*/config.yaml")),
        key=lambda c: c.agent,
    )
