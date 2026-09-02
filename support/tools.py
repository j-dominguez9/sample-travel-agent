"""Tools for the post-booking support agent.

The second agent, and the reason it exists: the customer said this travel agent
is "one of many rolling out in the next year", so the evaluation framework has
to work for an agent it was not written against. Anything in `evals/core` that
turns out to need changing to support this one was never framework code — it
was travel-agent code living in the wrong directory.

Deliberately a different shape from the travel agent: opaque booking references
rather than city names, a policy lookup rather than a search, and results that
are single records rather than lists. An agent-two that is a copy of agent-one
proves nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent.parent / "data" / "bookings.json"

_raw = json.loads(DATA.read_text())
BOOKINGS: list[dict[str, Any]] = _raw["bookings"]
POLICIES: list[dict[str, Any]] = _raw["policies"]


def lookup_booking(reference: str) -> dict[str, Any]:
    """Find one booking by its reference."""
    wanted = str(reference or "").strip().upper()
    for booking in BOOKINGS:
        if booking["reference"].upper() == wanted:
            return booking
    # A miss is a fact about their booking, not an error to narrate. The reply
    # evaluator checks the agent says so rather than inventing a record.
    return {"error": f"No booking found for reference {wanted}"}


def cancellation_policy(fare_class: str) -> dict[str, Any]:
    """Refund and change rules for a fare class."""
    wanted = str(fare_class or "").strip().lower()
    for policy in POLICIES:
        if policy["fare_class"].lower() == wanted:
            return policy
    return {"error": f"No policy on file for fare class {fare_class!r}"}


TOOL_FUNCTIONS = {
    "lookup_booking": lookup_booking,
    "cancellation_policy": cancellation_policy,
}

TOOLS = [
    {
        "name": "lookup_booking",
        "description": "Look up a booking by its six-character reference.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reference": {"type": "string", "description": "Booking reference, e.g. PLM4TQ"},
            },
            "required": ["reference"],
        },
    },
    {
        "name": "cancellation_policy",
        "description": "Get the refund and change rules for a fare class.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fare_class": {
                    "type": "string",
                    "description": "Fare class, e.g. Economy Saver, Flexible, Business",
                },
            },
            "required": ["fare_class"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> Any:
    try:
        return TOOL_FUNCTIONS[name](**tool_input)
    except Exception as exc:  # noqa: BLE001 — same contract as the travel agent
        return {"error": str(exc)}
