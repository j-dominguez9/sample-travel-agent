"""Evaluators that belong to no particular agent.

The reusability question the customer asked — "this is one of many agents this
year" — has a sharper form than "can you copy the directory". It is: when agent
two arrives, how much of what you wrote for agent one do you write again?

Three evaluators answer it, because none of them depend on the domain:

* **PII** is a property of text, not of travel. The same patterns, the same
  fail-closed behaviour, whatever the agent sells.
* **Hallucination** and **tool-response handling** are properties of the
  relationship between a tool result and a reply. Phoenix's pre-builts already
  express that generically.

What each agent still supplies is an **adapter**: a function turning its own
turn shape into the strings a judge expects. That is the seam. The prompt and
the domain are per tenant; the mechanism is not, and a new agent registers
these in three lines rather than reimplementing them.

Evaluators that encode domain truth — a flight number that must exist in the
fixtures, a booking reference that must have been returned — stay with their
agent, because there is nothing to share.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from phoenix.evals import LLM, bind_evaluator, create_evaluator
from phoenix.evals.metrics import HallucinationEvaluator, ToolResponseHandlingEvaluator

from evals.core.registry import Registry

Adapter = Callable[[Mapping[str, Any]], str]


def register_pii_invariant(registry: Registry, agent: str) -> Any:
    """Register `no_unredacted_pii` for `agent`.

    Checks the redaction control rather than the agent: if personal data is
    visible here, it reached the observability backend, which is a failure of
    the control regardless of which agent produced the turn. A privacy control
    that silently stops working is worse than none, because you believe you are
    covered.
    """

    @registry.register(
        agent=agent, name="no_unredacted_pii", kind="code", mode="invariant",
        online=True,
        description="No personal data reached the observability backend.",
    )
    @create_evaluator(name="no_unredacted_pii", kind="code")
    def no_unredacted_pii(input: Any, output: Any) -> dict:
        from common.redaction import PATTERNS

        parts = [
            str((input or {}).get("message", "")) if isinstance(input, dict)
            else str(input or "")
        ]
        if isinstance(output, dict):
            parts.append(str(output.get("reply") or ""))
            for call in output.get("tool_calls") or []:
                parts += [str(call.get("input")), str(call.get("output"))]
        blob = " ".join(parts)

        hits = {label: len(pattern.findall(blob)) for label, pattern in PATTERNS}
        found = {k: v for k, v in hits.items() if v}
        if not found:
            return {"score": 1.0, "label": "clean"}
        # Categories and counts, never the matched text: an explanation is
        # written back to Phoenix as an annotation, so quoting the match would
        # re-leak the data this evaluator exists to catch, into a second place.
        summary = ", ".join(f"{n}x {k}" for k, n in sorted(found.items()))
        return {
            "score": 0.0,
            "label": "unredacted_pii",
            "explanation": f"redaction appears to have failed: {summary} (values withheld)",
        }

    return no_unredacted_pii


def register_grounding_judges(
    registry: Registry,
    agent: str,
    *,
    llm: LLM,
    transcript: Adapter,
    reply: Adapter,
    tool_call: Adapter,
    tool_result: Adapter,
    applies: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Register the two generic judges for `agent`, bound to its adapters.

    The judges are Phoenix pre-builts and identical across tenants. Everything
    agent-specific arrives through the adapters, which is what stops this from
    being a copy-paste of one agent's `judges.py` into the next.
    """
    hallucination = bind_evaluator(
        HallucinationEvaluator(llm=llm),
        {"input": transcript, "output": reply},
    )
    tool_response_handling = bind_evaluator(
        ToolResponseHandlingEvaluator(llm=llm),
        {"input": transcript, "tool_call": tool_call,
         "tool_result": tool_result, "output": reply},
    )

    common = {
        "agent": agent, "kind": "llm", "mode": "signal", "suite": "capability",
        # Judges read only the request, the tool results and the reply — never a
        # golden label. That is what lets them run against live traffic, where
        # no labels exist.
        "online": True, "applies": applies,
    }
    registry.register(
        name="hallucination", direction="minimize",
        description="Claims unsupported by the tool results (1.0 = hallucinated).",
        **common,
    )(hallucination)
    registry.register(
        name="tool_response_handling", direction="maximize",
        description="Did the reply actually use what the tool returned?",
        **common,
    )(tool_response_handling)

    return {"hallucination": hallucination,
            "tool_response_handling": tool_response_handling}
