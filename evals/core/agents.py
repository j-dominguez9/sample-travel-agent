"""Load an agent's eval modules by name.

Registration is an import side effect: `evals/agents/<agent>/evaluators.py`
populates the registry when imported, and nothing is scoreable until it has
been. The DAGs used to do that with a literal `from evals.agents.travel import
evaluators, judges`, which was invisible while there was one agent and wrong
the moment there were two — the `support` DAG would have imported the travel
agent's evaluators, then asked the registry for `support`, found it empty, and
scored nothing at all. No error, no annotations, a green run.

That is the failure mode this module exists to remove: the DAG names the agent
it is building for, and the import follows from the name.

`judges.py` and `truth.py` are optional. An agent whose checks are all
deterministic has no judges module, and one with no computable ground truth has
no truth module — both are legitimate, so a missing file is not an error.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

_REQUIRED = ("evaluators",)
_OPTIONAL = ("judges", "truth")


@dataclass(frozen=True)
class AgentModules:
    evaluators: ModuleType
    judges: ModuleType | None = None
    truth: ModuleType | None = None


def load(agent: str) -> AgentModules:
    """Import `agent`'s eval modules, registering its evaluators as a side effect.

    Raises if the agent has no `evaluators` module: a config with no evaluators
    is a misconfiguration, and failing loudly beats a sweep that reports zero
    findings because there was nothing to find with.
    """
    loaded: dict[str, ModuleType | None] = {}
    for name in _REQUIRED:
        loaded[name] = importlib.import_module(f"evals.agents.{agent}.{name}")
    for name in _OPTIONAL:
        module_path = f"evals.agents.{agent}.{name}"
        try:
            loaded[name] = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            # Only "this agent has no judges.py" is absence. A ModuleNotFoundError
            # raised *inside* judges.py — a missing third-party package, a typo in
            # one of its own imports — must propagate: swallowing it yields
            # judges=None and an unregistered evaluator set, which is precisely
            # the green-run-that-scored-nothing this module exists to prevent.
            if exc.name != module_path:
                raise
            loaded[name] = None
    return AgentModules(**loaded)  # type: ignore[arg-type]
