"""Ground truth for the travel agent, derived from the fixtures.

The whole eval strategy rests on this module. Because every tool is backed by a
local JSON fixture, the correct answer to any query is *computable* rather than
a matter of judgment. That is what lets us build a labelled golden dataset with
no SME and no LLM in the labelling path — which in turn is what keeps the LLM
judges honest, since nothing an LLM generated is ever used as a label.

Everything here reads the same fixtures the agent's tools read, but reimplements
the lookup independently. Calling `agent.tools.search_flights` to produce the
expected answer would make the eval tautological: a bug in the tool would show
up identically on both sides and score as a pass.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

FLIGHTS = json.loads((DATA_DIR / "flights.json").read_text())
HOTELS = json.loads((DATA_DIR / "hotels.json").read_text())
WEATHER = json.loads((DATA_DIR / "weather.json").read_text())


# --------------------------------------------------------------------------
# Projections — the exact field set each tool hands back
#
# The fixtures carry more than the tools expose: a flight row has `origin`,
# `destination` and its availability window, but `search_flights` returns only
# five fields. Anything built for a judge must use these projections, or the
# judge is validated against context it will never have in production — which
# makes its measured accuracy meaningless for live traffic.
# --------------------------------------------------------------------------

FLIGHT_FIELDS = ("airline", "flight_number", "depart_time", "arrive_time", "price_usd")
HOTEL_FIELDS = ("name", "city", "price_per_night_usd", "rating")


def project_flight(f: dict) -> dict:
    return {k: f[k] for k in FLIGHT_FIELDS}


def project_hotel(h: dict) -> dict:
    return {k: h[k] for k in HOTEL_FIELDS}


# --------------------------------------------------------------------------
# Independent reimplementations of what each tool should return
# --------------------------------------------------------------------------

def expected_flights(origin: str, destination: str, on: str) -> list[dict]:
    """Flights that genuinely operate origin -> destination on `on`.

    Direction-sensitive on purpose: a reverse-leg flight is not an answer to an
    outbound query, and the set-comparison bug in `main` is exactly this.
    """
    out = []
    for f in FLIGHTS:
        if f["origin"].lower() != origin.lower():
            continue
        if f["destination"].lower() != destination.lower():
            continue
        if not (f["available_from"] <= on <= f["available_to"]):
            continue
        out.append(f)
    return out


def expected_hotels(city: str, check_in: str) -> list[dict]:
    return [
        h
        for h in HOTELS
        if h["city"].lower() == city.lower()
        and h["available_from"] <= check_in <= h["available_to"]
    ]


def has_weather(city: str) -> bool:
    return any(k.lower() == city.lower() for k in WEATHER)


def expected_itinerary_days(num_days: int) -> list[int]:
    """A trip of N days has days 1..N. The off-by-one in `main` drops day N."""
    return list(range(1, int(num_days) + 1))


# --------------------------------------------------------------------------
# Coverage maps — used by the generator to pick cases on both sides of a
# boundary rather than guessing which queries are answerable
# --------------------------------------------------------------------------

def routes() -> list[tuple[str, str]]:
    return sorted({(f["origin"], f["destination"]) for f in FLIGHTS})


def route_exists(origin: str, destination: str) -> bool:
    return (origin, destination) in set(routes())


def cities() -> dict[str, set[str]]:
    """Which cities each tool actually has data for."""
    return {
        "flight_origins": {f["origin"] for f in FLIGHTS},
        "flight_destinations": {f["destination"] for f in FLIGHTS},
        "hotels": {h["city"] for h in HOTELS},
        "weather": set(WEATHER),
    }


def route_window(origin: str, destination: str) -> tuple[str, str] | None:
    """Union of operating windows across every flight on a route."""
    legs = [
        f
        for f in FLIGHTS
        if f["origin"].lower() == origin.lower()
        and f["destination"].lower() == destination.lower()
    ]
    if not legs:
        return None
    return (
        min(f["available_from"] for f in legs),
        max(f["available_to"] for f in legs),
    )


def date_inside(origin: str, destination: str) -> str | None:
    """A date on which at least one flight on the route operates."""
    win = route_window(origin, destination)
    if win is None:
        return None
    lo, hi = (date.fromisoformat(w) for w in win)
    return (lo + (hi - lo) / 2).isoformat()


def date_outside(origin: str, destination: str) -> str | None:
    """A date on which the route exists but *no* flight operates.

    Returns None when every leg runs to the end of the fixture horizon — for
    those routes there is no out-of-window case to test, and inventing one
    would mean asserting behaviour the fixtures don't actually specify.
    """
    win = route_window(origin, destination)
    if win is None:
        return None
    hi = date.fromisoformat(win[1])
    if hi >= date(2026, 12, 31):
        return None
    return hi.replace(day=min(hi.day, 28)).replace(year=hi.year + 1).isoformat()


#: Travel dates must be in the future or the agent reasonably balks at the
#: premise, which would confound every score with an artefact of the fixture.
NOT_BEFORE = date(2026, 9, 1)


def date_full(origin: str, destination: str) -> str | None:
    """A future date on which *every* leg of the route operates."""
    legs = _legs(origin, destination)
    if not legs:
        return None
    lo = max([*(f["available_from"] for f in legs), NOT_BEFORE.isoformat()])
    hi = min(f["available_to"] for f in legs)
    if lo > hi:
        return None
    a, b = date.fromisoformat(lo), date.fromisoformat(hi)
    return (a + (b - a) / 2).isoformat()


def date_partial(origin: str, destination: str) -> str | None:
    """A future date on which some legs operate and others do not.

    The highest-value flight case in the suite: the same route returns a
    strictly smaller result set than `date_full` does, so an agent that ignores
    the date parameter — as `main` does — returns the full set and scores wrong.
    """
    legs = _legs(origin, destination)
    if len(legs) < 2:
        return None
    # Probe both sides of every window boundary — the day a seasonal route stops
    # and the day before one starts are both splitting dates — then take the
    # earliest, so cases land on plausible near-future travel dates.
    candidates = set()
    for f in legs:
        for boundary in (f["available_from"], f["available_to"]):
            b = date.fromisoformat(boundary)
            for offset in (-1, 0, 1):
                candidates.add(b.fromordinal(b.toordinal() + offset))
    splitting = [
        c.isoformat()
        for c in sorted(candidates)
        if c >= NOT_BEFORE and 0 < len(expected_flights(origin, destination, c.isoformat())) < len(legs)
    ]
    return splitting[0] if splitting else None


def _legs(origin: str, destination: str) -> list[dict]:
    return [
        f
        for f in FLIGHTS
        if f["origin"].lower() == origin.lower()
        and f["destination"].lower() == destination.lower()
    ]


def partial_window_routes() -> list[tuple[str, str]]:
    """Routes where some legs operate on a date and others don't.

    These are the most valuable flight cases: the same route returns different
    result sets depending on the date, so an agent that ignores the date
    parameter scores wrong on one of the two.
    """
    out = []
    for origin, destination in routes():
        legs = [
            f
            for f in FLIGHTS
            if f["origin"] == origin and f["destination"] == destination
        ]
        if len({(f["available_from"], f["available_to"]) for f in legs}) > 1:
            out.append((origin, destination))
    return out


def derive_expected(record: dict) -> tuple[dict, bool, str] | None:
    """This agent's ground truth for a curated candidate, or None.

    The framework hook (`evals.core.curate.derive_expected`) handles the cases
    that are true of any agent; this handles the two travel tools whose correct
    answer the fixtures can compute. Returning None means "I cannot derive this
    one", and the caller falls back to human review — which is the right
    default, because a wrong label in the golden set is worse than no label.
    """
    from evals.core.curate import is_iso_date

    calls = (record.get("output") or {}).get("tool_calls") or []
    if not calls:
        return None
    call = calls[0]
    name, args = call.get("name"), (call.get("input") or {})

    if name == "search_flights" and {"origin", "destination", "date"} <= set(args):
        # `expected_flights` compares ISO strings lexicographically and never
        # raises, so a malformed date silently matches nothing. Deriving from
        # that would publish "there are no flights on this route" as ground
        # truth when the real defect was the date the agent sent.
        if not is_iso_date(args["date"]):
            return ({}, True, f"agent sent a malformed date ({args['date']!r})")
        flights = expected_flights(args["origin"], args["destination"], args["date"])
        return (
            {
                "expected_tool": name,
                "expected_args": args,
                "expected_result_ids": sorted(f["flight_number"] for f in flights),
                "expected_behavior": "answer" if flights else "empty",
            },
            False,
            "derived from fixtures",
        )

    if name == "search_hotels" and {"city", "check_in"} <= set(args):
        if not is_iso_date(args["check_in"]):
            return ({}, True, f"agent sent a malformed date ({args['check_in']!r})")
        hotels = expected_hotels(args["city"], args["check_in"])
        return (
            {
                "expected_tool": name,
                "expected_args": args,
                "expected_result_ids": sorted(h["name"] for h in hotels),
                "expected_behavior": "answer" if hotels else "empty",
            },
            False,
            "derived from fixtures",
        )

    return None
