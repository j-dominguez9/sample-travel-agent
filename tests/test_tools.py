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
