"""Register each evaluator with Phoenix as an annotation config.

Phoenix (OSS) has no server-side evaluator registry — evaluators are code, and
the platform stores their *results*. Annotation configs are the one place an
evaluator gets a platform-side identity: its label set and, crucially, its
optimization direction. Without a config, a `hallucination` score of 1.0 is
just a number, and any chart built on it trends the wrong way.

Driven off the same registry as everything else, so a new agent's evaluators
appear in the platform by declaring them once in code.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from evals.core.registry import REGISTRY

BASE = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")


def _request(method: str, path: str, payload: dict | None = None) -> Any:
    req = urllib.request.Request(
        f"{BASE}{path}",
        method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        body = resp.read()
    return json.loads(body) if body else None


def existing() -> dict[str, str]:
    data = _request("GET", "/v1/annotation_configs") or {}
    return {c["name"]: c["id"] for c in data.get("data", [])}


def sync(agent: str, *, dry_run: bool = False) -> list[str]:
    """Create an annotation config for every registered evaluator."""
    have = existing()
    created = []
    for reg in REGISTRY.for_agent(agent):
        if reg.name in have:
            continue
        # Both tiers are binary pass/fail by design — the framework's guidance is
        # binary over Likert, because the criteria stay checkable.
        pass_label, fail_label = ("pass", "fail")
        if reg.direction == "minimize":
            # 1.0 is the bad outcome here (e.g. hallucinated), so the labels and
            # the direction both have to invert or the UI reads backwards.
            pass_label, fail_label = ("flagged", "clean")
        payload = {
            "name": reg.name,
            "type": "CATEGORICAL",
            "description": reg.description[:250] or f"{reg.kind} evaluator ({reg.mode})",
            "optimization_direction": reg.direction.upper(),
            "values": [
                {"label": pass_label, "score": 1.0},
                {"label": fail_label, "score": 0.0},
            ],
        }
        if dry_run:
            created.append(reg.name)
            continue
        _request("POST", "/v1/annotation_configs", payload)
        created.append(reg.name)
    return created


def delete(name: str) -> None:
    cfg = existing().get(name)
    if cfg:
        _request("DELETE", f"/v1/annotation_configs/{cfg}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from evals.agents.travel import evaluators, judges  # noqa: F401  (registers)

    delete("__probe__")
    agent = sys.argv[1] if len(sys.argv) > 1 else "travel"
    made = sync(agent)
    print(f"created {len(made)} annotation configs: {', '.join(made) or '(none new)'}")
    for name, cid in sorted(existing().items()):
        print(f"  {name:26} {cid}")
