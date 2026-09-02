"""Evaluators for the post-booking support agent — agent two.

Written to answer the customer's reusability question with a measurement rather
than an assertion. Of what this agent needs:

* PII and the two grounding judges come from `evals.core.library` unchanged;
  this file supplies adapters, not implementations.
* Three evaluators are genuinely new, because they encode this domain's truth —
  a booking reference that must have been returned by a tool, a fee that must
  match the policy on file, and the fact that this agent cannot transact either.

Everything else — sweep, thresholds, curation, diagnosis, cost, both DAGs —
was reused with no change at all. The DAG picked this agent up because a
config file exists, not because anything was added to it.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from phoenix.evals import LLM, create_evaluator

from evals.core.library import register_grounding_judges, register_pii_invariant
from evals.core.registry import REGISTRY

AGENT = "support"

_llm = LLM(provider="anthropic", model=os.getenv("JUDGE_MODEL", "claude-haiku-4-5"))

#: Booking references are six characters of upper-case letters and digits.
#: Two ways to recognise one, because neither alone is enough:
#:
#: * containing a digit — the shape of every reference in the fixtures, and
#:   unambiguous enough to spot unprompted;
#: * introduced as one ("booking PLMTQX", "reference RX9VBD") — which catches
#:   the all-alphabetic references a digit test misses entirely.
#:
#: The digit test cannot simply be dropped: a bare `[A-Z0-9]{6}` also matches
#: REFUND, CANCEL and STATUS, and this evaluator pages. A false positive here
#: wakes someone up for a bolded word.
_REF_BODY = r"[A-Z0-9]{6}"
REFERENCE = re.compile(
    rf"\b(?=[A-Z0-9]*\d){_REF_BODY}\b"
    rf"|(?:booking|reference|ref|PNR|record locator)[\s:#]+({_REF_BODY})\b",
    re.IGNORECASE,
)


def _references(text: str) -> set[str]:
    """Every booking reference mentioned in `text`, upper-cased."""
    found = set()
    for match in REFERENCE.finditer(text or ""):
        token = match.group(1) or match.group(0)
        found.add(token.upper())
    return found


def _calls(output: Any, name: str | None = None) -> list[dict]:
    calls = (output or {}).get("tool_calls", []) if isinstance(output, dict) else []
    return [c for c in calls if name is None or c.get("name") == name]


def _reply(row: Any) -> str:
    return ((row or {}).get("output") or {}).get("reply") or ""


def transcript(row: Any) -> str:
    """One turn as the conversation the assistant actually had.

    The travel agent's adapter renders flights and hotels; this one renders
    bookings and policies. Same function signature, same judge, different
    domain — which is the seam the framework is built around.
    """
    inp = (row or {}).get("input") or {}
    out = (row or {}).get("output") or {}
    lines = [f"User: {inp.get('message', '')}"]
    for call in out.get("tool_calls") or []:
        args = json.dumps(call.get("input"), sort_keys=True)
        result = json.dumps(call.get("output"), sort_keys=True)
        lines.append(f"Tool ({call.get('name')} {args}): {result}")
    return "\n".join(lines)


def _all_calls(row: Any, field: str) -> str:
    """Every call's `field` — never just the first.

    Agent one lost 26 turns to an adapter that showed the judge only call #1,
    making facts sourced from later calls look invented. Agent two starts with
    the fixed version; that is what reuse is supposed to buy.
    """
    calls = ((row or {}).get("output") or {}).get("tool_calls") or []
    if not calls:
        return ""
    if len(calls) == 1:
        return json.dumps(calls[0].get(field), sort_keys=True)
    return json.dumps(
        [{"tool": c.get("name"), field: c.get(field)} for c in calls],
        sort_keys=True,
    )


def _has_results(rec: Any) -> bool:
    return any(c.get("output") for c in _calls((rec or {}).get("output")))


# --------------------------------------------------------------------------
# Reused unchanged from the shared library
# --------------------------------------------------------------------------

no_unredacted_pii = register_pii_invariant(REGISTRY, AGENT)

_judges = register_grounding_judges(
    REGISTRY, AGENT,
    llm=_llm,
    transcript=transcript,
    reply=_reply,
    tool_call=lambda r: _all_calls(r, "input"),
    tool_result=lambda r: _all_calls(r, "output"),
    applies=_has_results,
)
hallucination = _judges["hallucination"]
tool_response_handling = _judges["tool_response_handling"]


# --------------------------------------------------------------------------
# This agent's own truth
# --------------------------------------------------------------------------

@REGISTRY.register(
    agent=AGENT, name="no_fabricated_reference", kind="code", mode="invariant",
    online=True,
    description="Every booking reference in the reply came back from a tool. "
                "The support equivalent of no_fabricated_flights — same idea, "
                "different identifier, so it cannot be shared.",
)
@create_evaluator(name="no_fabricated_reference", kind="code")
def no_fabricated_reference(input: Any, output: Any) -> bool:
    reply = (output or {}).get("reply", "") if isinstance(output, dict) else ""
    calls = _calls(output, "lookup_booking")
    if not calls:
        return True

    # Grounded means the reference came from somewhere the model does not
    # control: a tool result, or the user's own message. NOT the tool *input* —
    # that is the model's own text, so seeding from it lets an invented
    # reference launder itself into legitimacy by being looked up and failing.
    # A reply may still echo a reference the user supplied, including one that
    # was not found ("I couldn't find ZZZ999"), which is correct behaviour.
    message = (input or {}).get("message", "") if isinstance(input, dict) else str(input or "")
    grounded = _references(message)
    for call in calls:
        result = call.get("output")
        if isinstance(result, dict) and result.get("reference"):
            grounded.add(str(result["reference"]).upper())
    return _references(reply) <= grounded


@REGISTRY.register(
    agent=AGENT, name="fees_match_policy", kind="code", mode="invariant",
    online=True,
    description="A dollar figure in the reply matches a fee the policy tool "
                "returned. Quoting the wrong change fee is the costly failure "
                "here: the traveller budgets for it and is wrong at the desk.",
)
@create_evaluator(name="fees_match_policy", kind="code")
def fees_match_policy(input: Any, output: Any) -> bool:
    reply = (output or {}).get("reply", "") if isinstance(output, dict) else ""
    calls = _calls(output, "cancellation_policy")
    if not calls:
        return True
    legitimate = {0}
    for call in calls:
        result = call.get("output")
        if isinstance(result, dict) and result.get("change_fee_usd") is not None:
            legitimate.add(int(result["change_fee_usd"]))
    cited = {int(m.replace(",", "")) for m in re.findall(r"\$([\d,]+)", reply)}
    return cited <= legitimate


@REGISTRY.register(
    agent=AGENT, name="no_internal_leak", kind="code", mode="signal",
    suite="capability", online=True,
    description="Replies must not reveal tools, systems or data sources. The "
                "rule agent one learned; applied here without relearning it.",
)
@create_evaluator(name="no_internal_leak", kind="code")
def no_internal_leak(output: Any) -> bool:
    reply = ((output or {}).get("reply") or "").lower() if isinstance(output, dict) else ""
    return not any(
        phrase in reply
        for phrase in ("in the system", "in our system", "the database", "our records",
                       "the tool", "lookup returned", "i searched", "no results")
    )
