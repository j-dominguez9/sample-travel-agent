"""Tier A — hermetic tool tests. No LLM, no network, no Phoenix.

Runs in milliseconds for nothing, so it gates every push rather than every PR.
These assertions would have caught all four defects that shipped in `main`:
wrong-direction flights, ignored dates, the itinerary off-by-one, and the
temperature conversion.

The comparisons are against `evals.agents.travel.truth`, which reimplements each
lookup independently from the same fixtures. Calling the agent's own tools to
produce the expected value would make every test tautological — a bug would
appear identically on both sides and score as a pass.
"""

from __future__ import annotations

import pytest

from agent.tools import (
    create_itinerary,
    execute_tool,
    get_weather,
    search_flights,
    search_hotels,
)
from evals.agents.travel import truth as T

# --------------------------------------------------------------------------
# Flight direction — `main` compared {origin, destination} as a SET, so a
# return leg answered an outbound query.
# --------------------------------------------------------------------------

BIDIRECTIONAL = [
    (o, d)
    for (o, d) in T.routes()
    if (d, o) in set(T.routes())
]


@pytest.mark.parametrize(("origin", "destination"), BIDIRECTIONAL, ids=lambda v: v.replace(" ", ""))
def test_flights_are_direction_specific(origin: str, destination: str) -> None:
    """Each leg returns only flights that actually fly that way."""
    on = "2026-09-20"
    got = {f["flight_number"] for f in search_flights(origin, destination, on)}
    legal = {f["flight_number"] for f in T._legs(origin, destination)}
    assert got <= legal, f"{origin}->{destination} returned flights that don't fly that route"


# --------------------------------------------------------------------------
# Date filtering — `main` accepted `date` and ignored it entirely.
# --------------------------------------------------------------------------

DISCRIMINATING = [
    (o, d, T.date_full(o, d), T.date_partial(o, d))
    for (o, d) in T.partial_window_routes()
    if T.date_full(o, d) and T.date_partial(o, d)
]


@pytest.mark.parametrize(
    ("origin", "destination", "full_date", "partial_date"),
    DISCRIMINATING,
    ids=[f"{o}-{d}".replace(" ", "") for o, d, _, _ in DISCRIMINATING],
)
def test_date_changes_the_result_set(
    origin: str, destination: str, full_date: str, partial_date: str
) -> None:
    """The same route on two dates must return different sets.

    This is the assertion an agent that ignores `date` cannot pass: it would
    return the full set on both dates.
    """
    on_full = {f["flight_number"] for f in search_flights(origin, destination, full_date)}
    on_partial = {f["flight_number"] for f in search_flights(origin, destination, partial_date)}
    assert on_partial < on_full, "date is not being applied — both dates returned the same set"


@pytest.mark.parametrize(("origin", "destination"), T.routes(), ids=lambda v: v.replace(" ", ""))
def test_flights_match_independent_truth(origin: str, destination: str) -> None:
    for on in filter(None, {T.date_full(origin, destination), T.date_partial(origin, destination)}):
        got = sorted(f["flight_number"] for f in search_flights(origin, destination, on))
        expected = sorted(f["flight_number"] for f in T.expected_flights(origin, destination, on))
        assert got == expected, f"{origin}->{destination} on {on}"


def test_malformed_date_is_an_explicit_error() -> None:
    """A bad date must not silently match nothing — that reads as 'sold out'."""
    result = execute_tool(
        "search_flights",
        {"origin": "New York", "destination": "Miami", "date": "March 12, 2026"},
    )
    assert isinstance(result, dict) and "error" in result


# --------------------------------------------------------------------------
# Itinerary — `main` used range(1, n), dropping the final day.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("num_days", [1, 2, 3, 5, 7, 14])
def test_itinerary_covers_every_day(num_days: int) -> None:
    days = [d["day"] for d in create_itinerary("Chicago", num_days)["days"]]
    assert days == T.expected_itinerary_days(num_days)


# --------------------------------------------------------------------------
# Weather — `main` applied a bogus C-to-F conversion to Fahrenheit values.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("city", sorted(T.WEATHER))
def test_weather_tracks_the_fixture(city: str) -> None:
    """Reported temperatures stay within the tool's own +/-2 jitter."""
    reported = get_weather(city, "2026-10-20")
    fixture = T.WEATHER[city]
    for field in ("high_f", "low_f"):
        assert abs(reported[field] - fixture[field]) <= 2, (
            f"{city} {field}: {reported[field]} vs fixture {fixture[field]}"
        )


def test_weather_for_unknown_city_is_an_error_not_a_guess() -> None:
    assert "error" in get_weather("Atlantis", "2026-10-20")


# --------------------------------------------------------------------------
# Hotels — date windows were already honoured; this pins the behaviour.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("city", sorted(T.cities()["hotels"]))
def test_hotels_match_independent_truth(city: str) -> None:
    check_in = "2026-10-05"
    got = sorted(h["name"] for h in search_hotels(city, check_in, "2026-10-09"))
    expected = sorted(h["name"] for h in T.expected_hotels(city, check_in))
    assert got == expected


def test_tools_expose_only_their_documented_fields() -> None:
    """The judges' grounding rules depend on this field set being exactly right.

    If a tool starts returning more, the prompt telling the agent what it knows
    becomes wrong and the grounding evaluations silently drift.
    """
    flight = search_flights("New York", "Miami", "2026-10-01")[0]
    assert set(flight) == set(T.FLIGHT_FIELDS)
    hotel = search_hotels("Miami", "2026-10-05", "2026-10-09")[0]
    assert set(hotel) == set(T.HOTEL_FIELDS)


# --------------------------------------------------------------------------
# capability disclosure guardrail
#
# The trigger is "has this conversation been shown bookable options", not "did
# the user use a booking word". The keyword version scored 18/18 on the traffic
# set and 0/7 on phrasings it had not seen; these tests deliberately use the
# phrasings it missed.
# --------------------------------------------------------------------------

FLIGHTS = [{"name": "search_flights", "output": [{"flight_number": "UA 512"}]}]
NO_FLIGHTS = [{"name": "search_flights", "output": []}]
WEATHER = [{"name": "get_weather", "output": {"high_f": 73}}]


def test_showing_options_triggers_the_disclosure_without_a_booking_word() -> None:
    """The case keyword matching could never reach."""
    from common.capabilities import DISCLOSURE, ensure_disclosure

    out = ensure_disclosure(
        "What flights go from New York to Chicago on October 8?",
        "Here are your options: United UA 512 departs 6:50am — $178.",
        tool_calls=FLIGHTS,
    )
    assert out.startswith(DISCLOSURE)
    assert "United UA 512" in out


def test_a_later_confirmation_needs_no_second_disclosure() -> None:
    """"I'll take it" after the conversation was already told."""
    from common.capabilities import ensure_disclosure

    reply = "United UA 512 it is — departs 6:50am."
    assert ensure_disclosure("I'll take it.", reply, tool_calls=[],
                             already_disclosed=True) == reply


def test_an_empty_search_does_not_trigger_it() -> None:
    """"No flights that day" cannot be mistaken for a held seat."""
    from common.capabilities import ensure_disclosure

    reply = "There are no flights from Denver to Miami on that date."
    assert ensure_disclosure("Flights Denver to Miami?", reply,
                             tool_calls=NO_FLIGHTS) == reply


def test_weather_is_not_bookable() -> None:
    from common.capabilities import ensure_disclosure

    reply = "Tokyo will be 73F and rainy."
    assert ensure_disclosure("Weather in Tokyo?", reply, tool_calls=WEATHER) == reply


def test_an_explicit_request_with_nothing_to_show_still_discloses() -> None:
    """The keyword path still earns its place when no search ran."""
    from common.capabilities import DISCLOSURE, ensure_disclosure

    out = ensure_disclosure("Can you change the name on my existing reservation?",
                            "That is handled by the airline.", tool_calls=[])
    assert out.startswith(DISCLOSURE)


def test_guardrail_leaves_an_existing_disclosure_alone() -> None:
    """No double-disclosing when the model already got it right."""
    from common.capabilities import ensure_disclosure

    reply = "I can't hold a flight or take a deposit. Here's what's available: UA 512, $178."
    assert ensure_disclosure("Hold that flight and take a deposit.", reply,
                             tool_calls=FLIGHTS) == reply


def test_guardrail_and_its_evaluator_cannot_disagree() -> None:
    """The monitor must pass anything the control produces.

    If `booking_limits_disclosed` failed the guardrail's own output, the
    invariant would page on every transactional turn forever.
    """
    from common.capabilities import ensure_disclosure
    from evals.agents.travel.evaluators import booking_limits_disclosed

    message = "What flights go from New York to Chicago?"
    guarded = ensure_disclosure(message, "Here are your options: UA 512 — $178.",
                                tool_calls=FLIGHTS)
    score = booking_limits_disclosed.evaluate(
        {"input": {"message": message}, "output": {"reply": guarded, "tool_calls": FLIGHTS}}
    )[0].score
    assert score == 1.0


def test_evaluator_fails_options_shown_with_no_disclosure_anywhere() -> None:
    from evals.agents.travel.evaluators import booking_limits_disclosed

    res = booking_limits_disclosed.evaluate({
        "input": {"message": "Flights to Chicago?", "turn_index": 1,
                  "disclosed_by_turn": None},
        "output": {"reply": "Here are your options: UA 512 — $178.",
                   "tool_calls": FLIGHTS},
    })[0]
    assert res.score == 0.0 and res.label == "undisclosed"


def test_evaluator_accepts_a_disclosure_made_earlier_in_the_session() -> None:
    """Turn 3 is covered by what was said at turn 1."""
    from evals.agents.travel.evaluators import booking_limits_disclosed

    res = booking_limits_disclosed.evaluate({
        "input": {"message": "Book me a hotel there for two nights.",
                  "turn_index": 3, "disclosed_by_turn": 1},
        "output": {"reply": "Here are the hotels in Chicago.", "tool_calls": []},
    })[0]
    assert res.score == 1.0 and res.label == "disclosed_earlier"


def test_a_disclosure_made_later_does_not_excuse_an_earlier_turn() -> None:
    """Being told at turn 4 does not help the user who acted at turn 2."""
    from evals.agents.travel.evaluators import booking_limits_disclosed

    res = booking_limits_disclosed.evaluate({
        "input": {"message": "Flights to Chicago?", "turn_index": 2,
                  "disclosed_by_turn": 4},
        "output": {"reply": "Here are your options: UA 512 — $178.",
                   "tool_calls": FLIGHTS},
    })[0]
    assert res.score == 0.0


def test_implying_the_booking_happened_fails_even_with_a_disclosure() -> None:
    """Worse than silence: the user acts on a seat that was never held."""
    from evals.agents.travel.evaluators import booking_limits_disclosed

    res = booking_limits_disclosed.evaluate({
        "input": {"message": "Hold that flight.", "turn_index": 1,
                  "disclosed_by_turn": 1},
        "output": {"reply": "I can't process payments directly, but you're all set on UA 512.",
                   "tool_calls": FLIGHTS},
    })[0]
    assert res.score == 0.0 and res.label == "implied_completion"
