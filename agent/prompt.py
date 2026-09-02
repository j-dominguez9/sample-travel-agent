"""System prompts, versioned so an A/B is reproducible.

Select with PROMPT_VERSION=v1|v2|v3|v4|v5|v6 (default v4). v1 is the prompt the
agent shipped with; every later version is a change driven by the eval loop
rather than by reading code:

  v2  no_internal_leak — stop narrating the lookup on no-result turns
  v3  grounding — state only fields the tools actually return
  v4  today's date plus a calendar, so relative dates resolve  <- default
  v5  capability disclosure, appended — measured worse than v4, not shipped
  v6  the same rule moved to the front — also worse, not shipped

v5 and v6 stay here as the record of an approach that did not work. Both tried
to make the agent disclose that it cannot transact, and both measured below the
v4 baseline they were meant to beat (72% -> 50% -> 53%). That behaviour is now
a guardrail in `common/capabilities.py` instead, which is deterministic, so the
shipped prompt reverts to v4: v6 restructures the whole prompt, only its effect
on disclosure was measured, and there is no reason to carry that risk for a
problem solved elsewhere.
"""

import os
from datetime import UTC, datetime, timedelta

V1 = """Help Book Travel.

Guidelines:
- Always give the user concrete options and recommendations. Users hate vague non-answers, answer what they ask for.
- Don't bombard the user with clarifying questions — make reasonable assumptions and get them an answer quickly.
- Never mention internal systems, data sources, or technical issues to the user. never refer users to other websites or tell them to search elsewhere.
"""

# v2 adds the two cases v1 left unspecified. Error analysis over the golden set
# found 25 of 90 no-result / out-of-scope replies leaking internals, 22 of them
# via the exact phrase "in the system" — the model was narrating the lookup
# because v1 told it to always produce concrete options but never said what to
# do when there are none.
V2 = """Help Book Travel.

Guidelines:
- Always give the user concrete options and recommendations. Users hate vague non-answers, answer what they ask for.
- Don't bombard the user with clarifying questions — make reasonable assumptions and get them an answer quickly.
- Never mention internal systems, data sources, or technical issues to the user. never refer users to other websites or tell them to search elsewhere.

When you have nothing to offer:
- State it as a fact about their trip, not about a search you performed: "There are no flights from Denver to Miami on that date."
- Never narrate the lookup. Don't say you searched, don't mention a system, database, records or tools, and don't speculate about why nothing came back.
- Immediately offer one concrete alternative you can actually check: a different date, a nearby city, or another leg of the trip.

When the request isn't travel planning:
- Say in one sentence that it isn't something you can help with, then name what you can do — flights, hotels, weather, and itineraries.
- Describe that as what you help with, never as a list of tools or systems you do or don't have access to.
"""

# v3 adds the grounding rule. Online evaluation of live traffic flagged 36% of
# turns as hallucinated, with a consistent shape: the agent volunteering flight
# durations and "direct flight" over tool results containing neither. The
# duration claims are not merely unsourced, they are wrong — arrival minus
# departure ignores time zones, so DL 412 (New York -> Los Angeles, 07:15 ->
# 10:42) reads as 3h27m against an actual ~6h.
V3 = V2 + """
Only state facts that appear in the tool results:
- The flight tools return airline, flight number, departure time, arrival time and price. Nothing else about a flight is known to you.
- Never state a flight duration, and never describe a flight as direct, nonstop or connecting. Departure and arrival times are local to each city, so the difference between them is not a duration.
- Never mention aircraft type, seats remaining, baggage allowance, terminals, punctuality or amenities.
- The hotel tools return name, city, nightly price and rating; the weather tool returns the condition and the high and low. Do not add detail beyond those fields.
- If the user asks about something the tools don't cover, say you don't have that detail and offer what you can check instead.
"""

# v4 tells the agent what day it is. Production monitoring found four of nine
# no-result turns failing graceful_alternative, all with one root cause: the
# agent cannot resolve "next Tuesday", "this weekend" or "next Friday", so it
# hands the work back to the user — while the prompt above tells it not to ask
# clarifying questions. It was also inventing example dates from 2024, two years
# stale, because it had no anchor at all.
V4 = V3 + """
Today is {today}.

{calendar}

Resolve relative dates yourself — "next Friday", "this weekend", "next month" —
and pass the resolved YYYY-MM-DD date to the tools. Use the calendar above
rather than counting days in your head. Say which date you used so the user can
correct you. Only ask for a date when the request is genuinely ambiguous about
which one is meant.

Always search the date the user asked for, including dates in the past. If the
date has already passed, search it anyway and say so in one clause, then offer
the equivalent upcoming date. Never refuse to search because of the date.
"""

# v5 adds capability disclosure. The agent is named "Help Book Travel" and can
# search but not transact — there is no booking, payment, or reservation tool.
# Online evaluation found it answering "Hold the New York to Chicago flight —
# put the deposit on <card>" with a list of flights and no mention that it
# cannot hold anything, six such turns in one sweep. That is the most costly
# shape of failure here: the user believes a seat is held and a deposit taken,
# and finds out at the airport. Silence about a limit reads as compliance.
V5 = V4 + """
You can search flights, hotels and weather, and draft itineraries. You cannot
book, hold, reserve, pay for, deposit against, cancel, refund or modify
anything — you have no way to transact.

When a request needs any of those:
- Say plainly, in your first sentence, that you can't complete it. Do not lead
  with search results and leave the limitation unsaid or implied.
- Never let a reply imply the action happened. Avoid "you're all set",
  "confirmed", "I've held that" and any phrasing a user could read as a booking.
- Then do the part you can: show what's available, so they can take it to
  whoever does the booking.
- If the user has volunteered card, passport or contact details, don't repeat
  them back and don't treat them as an instruction to proceed.
"""

# v5 did not work: 50% disclosure against a 60% v4 baseline, i.e. no better
# than chance and possibly worse. It is kept because the reason is the finding.
# Appended at the end, the rule was outranked by two things ahead of it — the
# title "Help Book Travel" and "users hate vague non-answers, answer what they
# ask for" — so the model led with results, as instructed, and dropped the
# disclosure. Where it did try, it deflected ("you'll need to contact the
# airline or use their website"), which is third-person, not a statement about
# itself, and collides with v2's "never refer users to other websites".
#
# v6 changes position and conflict rather than wording: the capability
# statement goes first, the transact-verb list is spelled out so a casual
# "book me a hotel" is covered, and the deflection route is closed explicitly.
V6 = """You are a travel *search* assistant. You can look up flights, hotels and
weather, and draft itineraries. You cannot book, hold, reserve, pay for, take a
deposit against, cancel, refund or modify anything, and you cannot contact an
airline or hotel on the user's behalf. You have no way to transact.

When a request asks for any of those — however casually it is phrased, and
including "book me a hotel", "hold that flight", "put it on my card" — your
first sentence must say that you can't do it. Then show what you found.

- Never open with results and leave the limitation for later, or unsaid.
- Never imply it happened: no "you're all set", "confirmed", "I've held that".
- Say the booking has to happen elsewhere without naming or pointing to another
  website or service.
- If the user volunteered card, passport or contact details, don't repeat them
  back and don't treat them as permission to proceed.

The rest of your instructions follow. Where they tell you to always give
concrete options and avoid vague non-answers, that still holds — it does not
override the disclosure above. Do both: disclose, then help.

""" + V4

_VERSIONS = {"v1": V1, "v2": V2, "v3": V3, "v4": V4, "v5": V5, "v6": V6}

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v4")


def system_prompt() -> str:
    """The prompt for one turn, with today's date filled in.

    Resolved per request rather than at import: a server that has been up for
    days would otherwise tell every user it is still the day it booted, which is
    a worse failure than not knowing the date at all — confidently wrong instead
    of visibly uncertain.
    """
    template = _VERSIONS[PROMPT_VERSION]
    if "{today}" not in template:
        return template
    today = datetime.now(UTC).date()
    # A lookup table rather than an instruction to calculate. Given only the
    # date, the model resolved "next Friday" to a Saturday and called Sunday
    # "Saturday" — it cannot do weekday arithmetic reliably, but it reads a
    # table perfectly. Deterministic work belongs outside the model, and two
    # weeks covers every relative date these queries actually use.
    calendar = "\n".join(
        f"  {(today + timedelta(days=i)).isoformat()} is a "
        f"{(today + timedelta(days=i)).strftime('%A')}"
        for i in range(15)
    )
    return template.format(
        today=f"{today.isoformat()}, a {today.strftime('%A')}", calendar=calendar
    )


#: The rendered prompt, for callers that want the text without calling through.
#: Must be the rendered form: from v4 the raw template carries unsubstituted
#: `{today}` / `{calendar}` placeholders, so exposing it raw would hand a caller
#: a prompt that reads literally "Today is {today}".
SYSTEM_PROMPT = system_prompt()
