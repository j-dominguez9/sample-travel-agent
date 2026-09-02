"""Tier-2 LLM judges for the travel agent.

Tier 1 (`evaluators.py`) is deterministic and gates CI. These judges cover only
what code provably cannot score — and we know exactly where that boundary is,
because the v1 -> v2 prompt experiment crossed it: nine code evaluators all
passed v1's reply

    "I searched for hotels in Austin ... no results are currently showing up in
     the system. This could mean: 1. Hotels aren't yet available ..."

except the leak check, and not one of them could say that v2's reply

    "There are no hotels available in Austin for those dates. Would you like me
     to check a different date range, or would San Antonio work?"

was a *better answer*. Helpfulness and grounding-beyond-identifiers live on a
spectrum; that is what these judges are for.

Two are Phoenix pre-builts used as-is, two are custom classifiers. All four run
on Anthropic, which is the customer's only approved provider.

Per the framework's own guidance these are **signals**: logged and trended,
never asserted per case. Judge output is non-deterministic, so one weak verdict
must not turn CI red — gate on the aggregate instead.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from phoenix.evals import LLM, bind_evaluator, create_classifier
from phoenix.evals.metrics import HallucinationEvaluator, ToolResponseHandlingEvaluator

from evals.core.registry import REGISTRY

AGENT = "travel"

#: The judge model is configured independently of the system under test. Start
#: cheap and let validation decide: the right judge is the cheapest model that
#: clears >80% accuracy and >70% TPR/TNR against the labelled set, not the most
#: expensive one available.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-haiku-4-5")
_llm = LLM(provider="anthropic", model=JUDGE_MODEL)


# --------------------------------------------------------------------------
# Adapters: our task output -> the shapes the evaluators expect
# --------------------------------------------------------------------------

def today_line() -> str:
    """The date context the agent itself was given, restated for the judge.

    Prompt v4 injects today's date into the system prompt and instructs the
    agent to search past dates rather than refuse them. The judges were never
    given the same context, and it cost us: on the 2026-09-02 sweep both
    `hallucination` and `tool_response_handling` failed the turn

        User: ... any flights from New York on August 7, 2026?
        Agent: That date is in the past, but here are the flights that were
               available on August 7, 2026. For an upcoming early August trip ...

    with the reasoning "August 7, 2026 is actually a future date". It is not —
    the sweep ran on September 2nd. The agent was right and the judges marked
    it wrong, because a judge with no clock cannot check a claim about the
    calendar and guesses from its training cutoff instead.

    This belongs in the transcript rather than in a template: it is not a
    grading instruction, it is part of the context the agent genuinely had, and
    a judge grading faithfulness should see exactly what the agent saw.
    """
    today = datetime.now(UTC).date()
    return f"(Context: today's date is {today.isoformat()}, a {today.strftime('%A')}.)"


def transcript(row: Mapping[str, Any]) -> str:
    """Render one turn as the conversation the assistant actually had access to.

    `HallucinationEvaluator` treats this as its source of truth, so the tool
    results must appear verbatim — that is what makes an invented price or a
    false "cheapest" claim detectable.
    """
    inp = row.get("input") or {}
    out = row.get("output") or {}
    lines = [today_line(), f"User: {inp.get('message', '')}"]
    for call in out.get("tool_calls") or []:
        args = json.dumps(call.get("input"), sort_keys=True)
        result = json.dumps(call.get("output"), sort_keys=True)
        lines.append(f"Tool ({call.get('name')} {args}): {result}")
    return "\n".join(lines)


def _reply(row: Mapping[str, Any]) -> str:
    return (row.get("output") or {}).get("reply") or ""


def _all_calls(row: Mapping[str, Any], field: str) -> str:
    """Every tool call's `field`, not just the first one.

    This started as `_first_call` and it was quietly wrong. A turn here is
    routinely several calls — "what's the weather while I'm there?" fans out to
    one `get_weather` per night, and a weekend request calls `search_hotels`
    and `search_flights` — and showing the judge only call #1 makes every fact
    sourced from calls #2..n look invented. On the 2026-09-02 sweep that read
    as the agent fabricating three days of Tokyo weather it had actually looked
    up, one call each:

        tool_result (call 1 only): {"date": "2026-05-04", "condition": "Rain"}
        judge: "the agent fabricated weather for THREE additional dates"

    Multi-tool turns scored 20% against 77.8% for single-tool ones, which is
    the signature of an adapter bug rather than a behaviour: the agent does not
    get worse when it calls two tools, the evidence does.
    """
    calls = (row.get("output") or {}).get("tool_calls") or []
    if not calls:
        return ""
    if len(calls) == 1:
        return json.dumps(calls[0].get(field), sort_keys=True)
    # Keep the call's name alongside its payload — with several tools in play
    # the judge otherwise cannot tell which result belongs to which request.
    return json.dumps(
        [{"tool": c.get("name"), field: c.get(field)} for c in calls],
        sort_keys=True,
    )


# --------------------------------------------------------------------------
# Pre-built judges
# --------------------------------------------------------------------------

#: Source of truth is the conversation itself (tool calls + results), which is
#: why this is the right pre-built for an agent — Faithfulness expects a
#: separately supplied RAG context we do not have. NOTE: direction is
#: *minimize*; 1.0 means hallucinated.
hallucination = bind_evaluator(
    HallucinationEvaluator(llm=_llm),
    {"input": transcript, "output": _reply},
)

#: Did the agent actually use what the tool handed back, rather than talking
#: past it? Catches "here are your options" over an empty result set.
tool_response_handling = bind_evaluator(
    ToolResponseHandlingEvaluator(llm=_llm),
    {
        # Same clock the transcript carries — this judge sees the raw message
        # rather than the rendered transcript, so it needs the date separately.
        "input": lambda r: f"{today_line()}\nUser: {(r.get('input') or {}).get('message', '')}",
        "tool_call": lambda r: _all_calls(r, "input"),
        "tool_result": lambda r: _all_calls(r, "output"),
        "output": _reply,
    },
)


# --------------------------------------------------------------------------
# Custom judges
# --------------------------------------------------------------------------

GRACEFUL_TEMPLATE = """You are grading a travel agent's reply for a request it could not fulfil.

<request>{{request}}</request>
<reply>{{reply}}</reply>

The agent found nothing to offer, or the request was outside travel planning.
Grade only how well it handled that, not whether finding nothing was correct.

CRITERIA:
"graceful" = states plainly that nothing is available (or that it can't help),
             AND names at least one specific next step the agent could itself
             perform — a named city, a named date, another leg, or the
             categories of travel work it does handle.
"unhelpful" = any of: narrates its own search or mentions systems, databases,
             records or tools; speculates about why nothing was found; sends
             the user to another website or service; offers no specific next
             step at all.

Phrasing an offer as a question is normal and does NOT make a reply unhelpful.
What matters is whether a concrete alternative is named. "Would San Antonio
work?" names one. "Could you provide different dates?" names none — it hands
the work back to the user.

EXAMPLES:
Reply: "There are no hotels in Austin for those dates. Would San Antonio or
Dallas work, or shall I try a different date range?" -> graceful
  (a question, but it names two specific alternatives)
Reply: "No results found. Could you provide different dates?" -> unhelpful
  (a question that names nothing and asks the user to do the thinking)
Reply: "I searched but no results are showing up in the system. This could mean
hotels aren't listed yet." -> unhelpful
Reply: "I can't help with exchange rates, but I can search flights, hotels,
weather and itineraries for you." -> graceful
Reply: "I don't have access to that tool. Try your bank's website." -> unhelpful

EDGE CASES:
- Names an alternative but also narrates the search -> unhelpful (the leak is
  disqualifying on its own)
- Declines briefly and names what it does handle -> graceful
- Names an alternative without explicitly saying nothing was found -> graceful
- Empty reply -> unhelpful

Answer (graceful/unhelpful):"""

graceful_alternative = bind_evaluator(
    create_classifier(
        name="graceful_alternative",
        prompt_template=GRACEFUL_TEMPLATE,
        llm=_llm,
        choices={"graceful": 1.0, "unhelpful": 0.0},
    ),
    {"request": lambda r: (r.get("input") or {}).get("message", ""), "reply": _reply},
)


RECOMMENDATION_TEMPLATE = """You are grading whether a travel agent's recommendation matches what the user asked for.

<request>{{request}}</request>
<tool_results>{{results}}</tool_results>
<reply>{{reply}}</reply>

The tool results are the complete set of real options available. Judge whether
the reply's recommendation is justified by them and responsive to the request.

The tool results carry a fixed set of fields and NOTHING ELSE is known. Treat
any field not present there as unavailable — not as unstated-but-true. A claim
about an attribute the results do not contain is ungrounded even when it sounds
reasonable, because the agent had no source for it.

CRITERIA:
"grounded" = every option presented appears in the tool results; every stated
             value matches the results exactly; any superlative claim
             ("cheapest", "earliest", "best value") is actually true of that
             set; the reply addresses the route, city or dates requested; and
             every factual claim is about a field the results actually contain.
"ungrounded" = any of: presents an option absent from the tool results; states a
             price, time or rating that differs from the results; makes a
             superlative claim that is false for that set; recommends something
             for a different route, city or date than requested; OR asserts any
             attribute the tool results do not contain — flight duration, direct
             vs connecting, aircraft type, seats remaining, baggage allowance,
             terminal, punctuality, amenities.

EXAMPLES:
Results: [{"flight_number":"DL 883","depart_time":"09:40","price_usd":214},
          {"flight_number":"B6 1029","depart_time":"15:20","price_usd":189}]
Reply: "JetBlue B6 1029 at $189 is the cheapest." -> grounded
Reply: "Delta DL 883 at $214 is the cheapest." -> ungrounded (false superlative)
Reply: "DL 883 departs at 6:15 AM." -> ungrounded (results say 09:40)
Reply: "DL 883 departs 09:40 for $214. It's a direct flight of about 3h15m."
    -> ungrounded (neither "direct" nor a duration appears in the results)
Reply: "DL 883 departs 09:40 for $214. Checked baggage is included."
    -> ungrounded (baggage is not a field in the results)

EDGE CASES:
- Empty tool results and the reply offers nothing -> grounded
- Reply describes options accurately but recommends none -> grounded
- Arithmetic over stated values is still a new claim: arrival minus departure is
  NOT a duration, because the results carry no time zone -> ungrounded
- Generic travel advice tied to no specific option -> ignore it when grading

Answer (grounded/ungrounded):"""

recommendation_grounded = bind_evaluator(
    create_classifier(
        name="recommendation_grounded",
        prompt_template=RECOMMENDATION_TEMPLATE,
        llm=_llm,
        choices={"grounded": 1.0, "ungrounded": 0.0},
    ),
    {
        "request": lambda r: (r.get("input") or {}).get("message", ""),
        # Every call's results, for the same reason as tool_response_handling:
        # a superlative ("the cheapest") is graded against the options actually
        # found, and half the options are not a fair test of it.
        "results": lambda r: _all_calls(r, "output"),
        "reply": _reply,
    },
)


# --------------------------------------------------------------------------
# Registration — all signals, none of them gate CI
# --------------------------------------------------------------------------

def _no_results(rec: Mapping[str, Any]) -> bool:
    """True when the turn had nothing to offer.

    `graceful_alternative` is only meaningful on these turns; run it on a
    successful search and it correctly reports "this found flights", which as
    an aggregate reads as a false alarm.

    Two ways a turn has nothing to offer:

    * tools ran and every one came back empty;
    * no tool ran because the request was outside travel planning at all
      ("do I need a visa?", "what's the baggage allowance?").

    The second used to be written as "no tool ran", and that swept in a third
    case it should never have: a follow-up answered from the conversation
    ("which of those is cheapest?"). Those call no tool either, and the judge
    kept saying so — "the agent's reply actually found and offered a flight
    option, so it did NOT fail to fulfill the request." Split by gate reason,
    the intended population passed at 84.6% while the leaked one sat at 65.7%
    and dragged the metric under its floor.

    `turn_index` is what separates them, which is why the agent span now
    carries it. An opening message with no tool call is a genuine out-of-scope
    request; a later one is nearly always resolvable from what came before. On
    older spans it is absent — those keep the previous behaviour rather than
    being silently dropped from the denominator.
    """
    calls = (rec.get("output") or {}).get("tool_calls") or []
    if calls:
        return all(not c.get("output") for c in calls)
    turn_index = (rec.get("input") or {}).get("turn_index")
    return turn_index is None or turn_index == 1


def _has_results(rec: Mapping[str, Any]) -> bool:
    calls = (rec.get("output") or {}).get("tool_calls") or []
    return any(c.get("output") for c in calls)


for _ev, _name, _suite, _dir, _applies, _desc in (
    (hallucination, "hallucination", "capability", "minimize", _has_results,
     "Claims unsupported by the tool results in this conversation (1.0 = hallucinated)."),
    (tool_response_handling, "tool_response_handling", "capability", "maximize", _has_results,
     "Did the reply actually use what the tool returned?"),
    (graceful_alternative, "graceful_alternative", "capability", "maximize", _no_results,
     "No-result and out-of-scope replies state the outcome and offer a real next step."),
    (recommendation_grounded, "recommendation_grounded", "regression", "maximize", _has_results,
     "Recommendations and superlatives are true of the options actually found."),
):
    REGISTRY.register(
        agent=AGENT, name=_name, kind="llm", mode="signal",
        suite=_suite,
        # Every judge reads only the request, the tool results, and the reply —
        # never a golden label. That is what lets them run against live traffic,
        # and it is the sharpest argument for Tier 2: in production there are no
        # labels, so most of the deterministic suite simply cannot run.
        online=True,
        direction=_dir,
        applies=_applies,
        description=_desc,
    )(_ev)
