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
    for call in _calls(output):
        args = call.get("input") or {}
        if not isinstance(args, dict):
            continue
        if wanted & {str(v) for k, v in args.items() if "date" in k or k in _DATE_KEYS}:
            return True
    return False


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
