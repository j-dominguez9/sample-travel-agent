"""Deterministic evaluators for the travel agent.

Every check here is code, not a judge: the fixtures make the correct answer
computable, so these cost nothing to run, never drift, and can gate CI hard.
The LLM judges in `judges.py` cover only what genuinely needs taste.

Each evaluator reads the task output produced by `evals.core.runner.run_turn`:

    {"reply": str, "tool_calls": [{"name": str, "input": dict, "output": Any}]}
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from datetime import date as date_cls
from typing import Any

from phoenix.evals import create_evaluator

from evals.agents.travel import truth as T
from evals.core.registry import REGISTRY

AGENT = "travel"

#: Flight numbers look like "DL 883" / "B6 1029"; used to spot fabricated ones.
FLIGHT_NUM = re.compile(r"\b([A-Z]{2}|[A-Z]\d)\s?(\d{2,4})\b")

#: Argument names that carry a date but are not spelled "date".
_DATE_KEYS = {"check_in", "check_out"}

#: Every carrier code the fixtures actually contain. A citation is only judged
#: when it uses one of these, so real prose can't trip the check.
CARRIER_CODES = {f["flight_number"].split()[0].upper() for f in T.FLIGHTS}


def _calls(output: Any, name: str | None = None) -> list[dict]:
    calls = (output or {}).get("tool_calls", []) if isinstance(output, dict) else []
    return [c for c in calls if name is None or c.get("name") == name]


def _norm_flight(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


# --------------------------------------------------------------------------
# Invariants — these gate CI
# --------------------------------------------------------------------------

@REGISTRY.register(
    agent=AGENT, name="tool_selection", kind="code", mode="invariant",
    description="The tool the query calls for is the tool that ran.",
)
@create_evaluator(name="tool_selection", kind="code")
def tool_selection(output: Any, expected: dict) -> bool:
    want = expected.get("expected_tool")
    called = {c["name"] for c in _calls(output)}
    if want is None:
        # Out-of-scope queries should not reach a tool at all.
        return len(called) == 0
    return want in called


@REGISTRY.register(
    agent=AGENT, name="date_grounding", kind="code", mode="invariant",
    description="A natural-language date reached the tool as the right ISO date.",
)
@create_evaluator(name="date_grounding", kind="code")
def date_grounding(output: Any, expected: dict) -> bool:
    want_date = expected.get("expected_args", {}).get("date")
    if not want_date:
        return True  # not a dated query; nothing to check
    for c in _calls(output, expected.get("expected_tool")):
        if (c.get("input") or {}).get("date") == want_date:
            return True
    return False


@REGISTRY.register(
    agent=AGENT, name="tool_result_exact", kind="code", mode="invariant",
    description="Tool output matches the fixtures exactly — catches direction "
                "and date-window bugs that a plausible-looking reply hides.",
)
@create_evaluator(name="tool_result_exact", kind="code")
def tool_result_exact(output: Any, expected: dict) -> bool:
    want = set(expected.get("expected_result_ids") or [])
    tool = expected.get("expected_tool")
    if tool not in {"search_flights", "search_hotels"}:
        return True
    for c in _calls(output, tool):
        res = c.get("output")
        if not isinstance(res, list):
            continue
        key = "flight_number" if tool == "search_flights" else "name"
        got = {str(r.get(key)) for r in res if isinstance(r, dict)}
        return got == want
    return not want  # tool never ran: correct only if nothing was expected


@REGISTRY.register(
    agent=AGENT, name="flight_direction", kind="code", mode="invariant", online=True,
    description="No returned flight travels the opposite way to the request. "
                "Needs no label — legality comes from the call's own arguments.",
)
@create_evaluator(name="flight_direction", kind="code")
def flight_direction(output: Any) -> bool:
    for c in _calls(output, "search_flights"):
        args, res = c.get("input") or {}, c.get("output")
        o, d = args.get("origin"), args.get("destination")
        if not (o and d and isinstance(res, list)):
            continue
        legal = {f["flight_number"] for f in T._legs(o, d)}
        for r in res:
            if isinstance(r, dict) and r.get("flight_number") not in legal:
                return False
    return True


@REGISTRY.register(
    agent=AGENT, name="itinerary_day_count", kind="code", mode="invariant",
    description="A trip of N days contains days 1..N — catches the off-by-one.",
)
@create_evaluator(name="itinerary_day_count", kind="code")
def itinerary_day_count(output: Any, expected: dict) -> bool:
    if expected.get("expected_tool") != "create_itinerary":
        return True
    want = [int(x) for x in expected.get("expected_result_ids") or []]
    for c in _calls(output, "create_itinerary"):
        res = c.get("output")
        if isinstance(res, dict):
            return [d.get("day") for d in res.get("days", [])] == sorted(want)
    return False


@REGISTRY.register(
    agent=AGENT, name="weather_values_plausible", kind="code", mode="invariant",
    online=True,
    description="Reported temperatures track the fixtures within the intended "
                "jitter — catches the bogus C-to-F conversion applied to values "
                "that were already Fahrenheit. Needs no label.",
)
@create_evaluator(name="weather_values_plausible", kind="code")
def weather_values_plausible(output: Any) -> bool:
    if not _calls(output, "get_weather"):
        return True
    for c in _calls(output, "get_weather"):
        res, args = c.get("output"), (c.get("input") or {})
        if not isinstance(res, dict) or "high_f" in res is None:
            continue
        entry = next(
            (v for k, v in T.WEATHER.items() if k.lower() == str(args.get("city", "")).lower()),
            None,
        )
        if entry is None:
            return "error" in res  # no data for that city is the correct answer
        # The tool applies at most +/-2 of deliberate jitter to each value.
        for field in ("high_f", "low_f"):
            got = res.get(field)
            if got is None or abs(got - entry[field]) > 2:
                return False
    return True


@REGISTRY.register(
    agent=AGENT, name="no_fabricated_flights", kind="code", mode="invariant", online=True,
    description="Every flight number in the reply appeared in a tool result. "
                "Deterministic hallucination detection — no judge needed.",
)
@create_evaluator(name="no_fabricated_flights", kind="code")
def no_fabricated_flights(output: Any) -> bool:
    reply = (output or {}).get("reply", "") if isinstance(output, dict) else ""
    grounded = {
        _norm_flight(str(r.get("flight_number")))
        for c in _calls(output, "search_flights")
        if isinstance(c.get("output"), list)
        for r in c["output"]
        if isinstance(r, dict) and r.get("flight_number")
    }
    # Only consider tokens carrying a carrier code that exists in the fixtures.
    # That excludes prose like "Terminal 4" or "Gate B12" without excluding a
    # fabricated flight on a real carrier, which is the case that matters:
    # "UA 999" is a plausible-looking invention and must be caught.
    if not _calls(output, "search_flights"):
        return True
    cited = {
        _norm_flight(f"{a}{b}")
        for a, b in FLIGHT_NUM.findall(reply)
        if a.upper() in CARRIER_CODES
    }
    return cited <= grounded


# --------------------------------------------------------------------------
# Signals — logged and trended, never asserted per case
# --------------------------------------------------------------------------

#: Phrases that expose the machinery behind the agent. The system prompt says
#: "Never mention internal systems, data sources, or technical issues to the
#: user", so any of these in a reply is a prompt-adherence failure. Both the
#: no-results path and the out-of-scope path leak, which is why this is checked
#: on both rather than only when a search came back empty.
LEAK_PHRASES = (
    "in the system", "in our database", "the database", "no data",
    "our records", "internal", "fixture", "api returned", "data source",
    "access to tools", "my tools", "the tool", "i don't have a tool",
    "not in my", "backend", "search returned",
)


@REGISTRY.register(
    agent=AGENT, name="no_internal_leak", kind="code", mode="signal",
    suite="capability", online=True,
    description="Replies on the no-results and out-of-scope paths must not "
                "reveal tools, systems, or data sources to the user.",
)
@create_evaluator(name="no_internal_leak", kind="code")
def no_internal_leak(output: Any) -> bool:
    # Applicability is inferable from the turn itself, with no golden label:
    # no tool ran at all (out of scope), or every tool came back empty. That is
    # what lets this run against live traffic, where labels do not exist.
    calls = _calls(output)
    empty = bool(calls) and all(not c.get("output") for c in calls)
    if calls and not empty:
        return True
    reply = ((output or {}).get("reply") or "").lower()
    return not any(leak in reply for leak in LEAK_PHRASES)


@REGISTRY.register(
    agent=AGENT, name="stays_in_scope", kind="code", mode="signal",
    suite="capability",
    description="Out-of-scope asks should not trigger a travel tool call.",
)
@create_evaluator(name="stays_in_scope", kind="code")
def stays_in_scope(output: Any, expected: dict) -> bool:
    if expected.get("expected_behavior") != "out_of_scope":
        return True
    return len(_calls(output)) == 0


# --------------------------------------------------------------------------
# From axial coding over 24 production traces
# (coding-run:travel-agent-failure-taxonomy-2026-09-01)
#
# Two categories in that taxonomy had no evaluator. Both are added here rather
# than as judges because both are decidable from the turn itself.
# --------------------------------------------------------------------------

#: "March 12, 2026" / "2026-03-12" — the two shapes real requests actually use.
_EXPLICIT_DATE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})\b"
    r"|\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


def _requested_dates(message: str) -> list[date_cls]:
    out = []
    for iso, month, day, year in _EXPLICIT_DATE.findall(message):
        try:
            out.append(date_cls.fromisoformat(iso) if iso
                       else date_cls(int(year), _MONTHS[month.lower()], int(day)))
        except ValueError:
            continue
    return out


@REGISTRY.register(
    agent=AGENT, name="past_date_still_searched", kind="code", mode="invariant",
    online=True,
    description="A date that has already passed is still searched. Trace coding "
                "found the agent refusing 'March 12, 2026' as past while serving "
                "'April 20, 2026' in the same run — same policy, opposite answers.",
)
@create_evaluator(name="past_date_still_searched", kind="code")
def past_date_still_searched(input: Any, output: Any) -> bool:
    message = (input or {}).get("message", "") if isinstance(input, dict) else str(input or "")
    past = [d for d in _requested_dates(message) if d < datetime.now(UTC).date()]
    if not past:
        return True
    # The prompt's rule: search it anyway and note that it has passed. A refusal
    # leaves the user with nothing, and refusing only sometimes is worse still.
    #
    # Compliance is the requested date reaching a tool — not merely that some
    # tool ran. An agent that declines a past-dated flight search and calls
    # get_weather has still refused the request, and `get_weather` takes a date,
    # so a tool-name check alone would score that as compliant.
    wanted = {d.isoformat() for d in past}
    dated_calls = []
    for call in _calls(output):
        args = call.get("input") or {}
        if not isinstance(args, dict):
            continue
        supplied = {str(v) for k, v in args.items() if "date" in k or k in _DATE_KEYS}
        if supplied:
            dated_calls.append(supplied)
            if wanted & supplied:
                return True

    # Not every tool accepts a date. `create_itinerary` takes destination and
    # num_days only, so "a 5-day itinerary for Paris, arriving June 10, 2026"
    # is fulfilled correctly with the date never reaching an argument — and an
    # earlier version of this check flagged exactly that as a refusal.
    #
    # A dated call that carries some *other* date is still a refusal: the agent
    # quietly searched a day the user did not ask for.
    if dated_calls:
        return {"score": 0.0, "label": "refused"}

    # No call took a date at all, so whether the past date was honoured is not
    # decidable from this turn. Scored None rather than 1.0: an unverifiable
    # turn counted as a pass inflates the very metric that gates on it, and
    # `sweep()` drops null scores, so this leaves the denominator instead of
    # padding it. Returning False would be worse still — `create_itinerary`
    # legitimately takes no date.
    if not _calls(output):
        return {"score": 0.0, "label": "refused"}
    return {"score": None, "label": "unverifiable"}


@REGISTRY.register(
    agent=AGENT, name="derived_totals_are_correct", kind="code", mode="invariant",
    online=True,
    description="A stated total equals a returned nightly rate times the nights "
                "booked. The largest taxonomy category was values computed from "
                "tool output and presented as fact — sound arithmetic is fine, "
                "wrong arithmetic reads exactly the same to a customer.",
)
@create_evaluator(name="derived_totals_are_correct", kind="code")
def derived_totals_are_correct(output: Any) -> bool:
    calls = _calls(output, "search_hotels")
    if not calls:
        return True
    reply = (output or {}).get("reply", "") if isinstance(output, dict) else ""

    # Rates and stay length are paired PER CALL. Pooling them was wrong twice
    # over: two searches with different date ranges made a total from the first
    # look fabricated, and one malformed call reset `nights` to 0, dropping
    # every legitimate total from the set.
    rates: set[int] = set()
    legitimate: set[int] = set()
    for call in calls:
        args, res = call.get("input") or {}, call.get("output")
        if not isinstance(res, list):
            continue
        call_rates = {int(h["price_per_night_usd"]) for h in res
                      if isinstance(h, dict) and "price_per_night_usd" in h}
        rates |= call_rates
        legitimate |= call_rates
        try:
            nights = (date_cls.fromisoformat(str(args["check_out"]))
                      - date_cls.fromisoformat(str(args["check_in"]))).days
        except (KeyError, TypeError, ValueError):
            continue  # this call contributes rates but no derivable total
        if nights > 0:
            legitimate |= {r * nights for r in call_rates}
    if not rates:
        return True
    cited = {int(m.replace(",", "")) for m in re.findall(r"\$([\d,]+)", reply)}
    # Ignore anything below the cheapest rate: ratings, counts and night tallies
    # are not money, and a total is never smaller than one night.
    return all(v in legitimate for v in cited if v >= min(rates))


@REGISTRY.register(
    agent=AGENT, name="no_unredacted_pii", kind="code", mode="invariant",
    online=True,
    description="No personal data reached the observability backend. This checks "
                "the redaction control itself, not the agent — a privacy control "
                "that silently stops working is worse than none, because you "
                "believe you are covered.",
)
@create_evaluator(name="no_unredacted_pii", kind="code")
def no_unredacted_pii(input: Any, output: Any) -> dict:
    from common.redaction import PATTERNS

    parts = [str((input or {}).get("message", "")) if isinstance(input, dict) else str(input or "")]
    if isinstance(output, dict):
        parts.append(str(output.get("reply") or ""))
        for call in output.get("tool_calls") or []:
            parts += [str(call.get("input")), str(call.get("output"))]
    blob = " ".join(parts)

    hits = {label: len(pattern.findall(blob)) for label, pattern in PATTERNS}
    found = {k: v for k, v in hits.items() if v}
    if not found:
        return {"score": 1.0, "label": "clean"}
    # Name the categories and counts, never the matched text. An explanation is
    # written back to Phoenix as an annotation, so quoting the match here would
    # re-leak the data this evaluator exists to catch — into a second place.
    summary = ", ".join(f"{n}x {k}" for k, n in sorted(found.items()))
    return {
        "score": 0.0,
        "label": "unredacted_pii",
        "explanation": f"redaction appears to have failed: {summary} (values withheld)",
    }


# --------------------------------------------------------------------------
# Capability disclosure
#
# The agent can search but not transact: there is no booking, payment or
# reservation tool behind it. Online evaluation found it answering
#
#     "Hold the New York to Chicago flight on October 8, 2026 — put the
#      deposit on <card>."
#
# with a list of flights and no mention that it cannot hold anything — six such
# turns in one sweep. Unlike a wrong price, this failure is invisible to the
# user at the time: they believe a seat is held and a deposit taken, and find
# out at the airport. That is why it is an invariant rather than a signal.
# --------------------------------------------------------------------------

# The patterns live in `common/capabilities.py`, next to the guardrail that
# uses them, and are imported rather than restated. Two copies would drift, and
# a monitor that has drifted from the control it watches reports health it
# cannot see — the same reason `no_unredacted_pii` imports PATTERNS from
# `common/redaction.py` instead of keeping its own list.
from common.capabilities import DISCLOSES as _DISCLOSES
from common.capabilities import FALSE_CONFIRMATION as _FALSE_CONFIRMATION
from common.capabilities import TRANSACT as _TRANSACT
from common.capabilities import presents_bookable_options as _presents_bookable_options


@REGISTRY.register(
    agent=AGENT, name="booking_limits_disclosed", kind="code", mode="invariant",
    online=True,
    description="By the time a user has been shown bookable options — or has "
                "asked outright to transact — the conversation must have told "
                "them the agent cannot book, and must never imply that it did.",
)
@create_evaluator(name="booking_limits_disclosed", kind="code")
def booking_limits_disclosed(input: Any, output: Any) -> dict:
    inp = input if isinstance(input, dict) else {}
    message = str(inp.get("message", "") if inp else input or "")
    out = output if isinstance(output, dict) else {}
    reply = out.get("reply") or ""
    calls = out.get("tool_calls") or []

    # Checked before applicability: a reply claiming the booking happened is a
    # failure whatever prompted it.
    if _FALSE_CONFIRMATION.search(reply):
        return {
            "score": 0.0,
            "label": "implied_completion",
            "explanation": "reply implies the booking or payment went through",
        }

    at_risk = _presents_bookable_options(calls) or bool(_TRANSACT.search(message))
    if not at_risk:
        return {"score": 1.0, "label": "not_applicable"}

    if _DISCLOSES.search(reply):
        return {"score": 1.0, "label": "disclosed"}

    # Said earlier in the same conversation still counts — `disclosed_by_turn`
    # is the first turn of this session whose reply disclosed. A user shown
    # flights at turn 1 and told then does not need telling again at turn 3.
    turn = inp.get("turn_index")
    disclosed_by = inp.get("disclosed_by_turn")
    if disclosed_by is not None and turn is not None and disclosed_by <= turn:
        return {"score": 1.0, "label": "disclosed_earlier"}
    # Single-turn callers (the offline experiment path) supply neither field,
    # so absence of session context falls back to judging this turn alone.
    return {
        "score": 0.0,
        "label": "undisclosed",
        "explanation": "user was shown bookable options, or asked to transact, "
                       "with no statement anywhere in the conversation that the "
                       "agent cannot book",
    }
