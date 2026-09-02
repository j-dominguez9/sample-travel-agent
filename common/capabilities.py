"""What the agent can and cannot do, enforced in code rather than asked for.

The agent can search; it cannot transact. There is no booking, payment or
reservation tool behind it. A user who reads "here are your options" as
confirmation that a seat is held and a deposit taken does not find out until
the airport, which makes silence about the limit the most expensive failure
this agent has.

Two prompt versions tried to fix it by instruction and both failed, measured on
the same six transactional conversations x3 passes:

    v4 (no rule)                72%  (n=25)
    v5 (rule appended)          50%
    v6 (rule moved to the top)  53%  (n=34)

The failures were not random. The agent disclosed reliably when it had nothing
to offer anyway, and went quiet exactly when a search succeeded — "Hold the New
York to Chicago flight ... put the deposit on <card>" failed 5 of 5 on v6,
answered with a clean list of flights and no mention that nothing was held. A
rule competing against "always give the user concrete options" loses whenever
there are concrete options to give.

So this is a guardrail, not a guideline. The disclosure is added by code when
the model omits it, which is the same choice already made for PII: a control
that matters is not left to a sampler. `booking_limits_disclosed` then plays
the part `no_unredacted_pii` plays for redaction — it monitors the control, and
its job is to notice if this stops working.

**What triggers it, and why it is not intent detection.**

The first version keyed off transactional vocabulary in the user's message, and
scored 18/18 on the traffic set — which turned out to mean nothing, because the
traffic set was written by the same person as the regex and always spelled
booking intent as "book" or "hold". Probed with phrasings it had not seen, it
failed 7/7 in each direction:

    "Hold on — what about Tuesday?"        fired, and shouldn't have
    "How much will I pay for a hotel?"     fired, and shouldn't have
    "I'll take it."                        missed, and is the dangerous one
    "Go ahead with the first option."      missed
    "Take the 6:50am one."                 missed

The misses are not a tuning problem. After the agent lists two flights, a
confirmation carries no transactional vocabulary at all — the intent lives in
the conversation, not the sentence — and that is precisely the moment a user
believes something was reserved.

So the trigger is no longer intent. The question this asks is the one the code
can actually answer: **has this conversation been shown bookable options?** If
it has, the user is at risk of thinking they are held, whatever they type next.
The disclosure goes out once, when options first appear, and covers every later
"I'll take it" because it was already said. Vocabulary is kept only as a second
path, for an explicit request that returns nothing to show — and it is no
longer load-bearing, so its false positives cost one redundant sentence rather
than a missed disclosure.
"""

from __future__ import annotations

import re

#: The vocabulary of transacting, as word families rather than bare stems.
#: `\bbook\b` does not match "bookings" and `\breserve\b` does not match
#: "reservations", and both forms occur in real replies.
_TXN = (
    r"book(?:s|ing|ings|ed)?|hold(?:s|ing)?|reserv(?:e|es|ed|ing|ation|ations)|"
    r"deposit(?:s)?|pay(?:s|ing|ment|ments)?|purchas(?:e|es|ed|ing)|buy(?:ing)?|"
    r"charg(?:e|es|ed|ing)|refund(?:s|ed)?|cancel(?:s|led|lation|lations)?|"
    r"rebook(?:ing)?|transact(?:ion|ions)?"
)

#: A request that asks the agent to *do* something rather than look it up.
#: Past tense counts: "I booked a flight through you last month and need a
#: refund" is a transactional request about a past booking.
TRANSACT = re.compile(
    rf"\b(?:{_TXN}|change (?:the )?(?:name|flight|booking|reservation))\b",
    re.IGNORECASE,
)

#: A first-person statement of inability. Bounded and sentence-local, so an
#: "I can't do that" several sentences from an unrelated "booking" cannot pair
#: up into a false pass.
DISCLOSES = re.compile(
    r"\b(?:i|we)\s+(?:can(?:no|')?t|cannot|am\s+not\s+able|'m\s+not\s+able|"
    r"am\s+unable|'m\s+unable|do\s+not|don't|have\s+no\s+way)\b"
    rf"[^.!?]{{0,90}}?\b(?:{_TXN}|process|complete|manage|make)\b"
    rf"|\b(?:unable|not\s+able)\s+to\s+(?:{_TXN}|complete|process)\b"
    rf"|\bno\s+(?:way|ability)\s+to\s+(?:{_TXN}|complete)\b",
    re.IGNORECASE,
)

#: Phrasing that asserts the transaction happened. Strictly worse than saying
#: nothing — the user acts on a seat that was never held.
FALSE_CONFIRMATION = re.compile(
    r"\byou'?re\s+(?:all\s+set|booked|confirmed)\b"
    r"|\b(?:i'?ve|i\s+have)\s+(?:booked|held|reserved|charged|processed)\b"
    r"|\byour\s+(?:booking|reservation|deposit)\s+is\s+confirmed\b",
    re.IGNORECASE,
)

#: Prepended, not appended: the point is that the user reads it before the
#: options, not after deciding. Phrased as what the agent is rather than an
#: apology, and it names no other website — v2 forbids sending users elsewhere,
#: and the model's own attempts at this ("contact the airline or use their
#: website") broke that rule.
DISCLOSURE = (
    "Just so you know — I can't book, hold, or pay for anything; "
    "I can only look things up and show you what's available."
)


#: Tools whose results a user could mistake for a reservation. Weather and
#: itineraries are not bookable, so seeing them creates no such impression.
BOOKABLE_TOOLS = frozenset({"search_flights", "search_hotels"})


def presents_bookable_options(tool_calls: list[dict] | None) -> bool:
    """True when this turn put flights or hotels in front of the user.

    Empty results don't count: "there are no flights that day" cannot be
    mistaken for a held seat.
    """
    for call in tool_calls or []:
        if call.get("name") not in BOOKABLE_TOOLS:
            continue
        result = call.get("output")
        if isinstance(result, list) and result:
            return True
    return False


def discloses(text: str) -> bool:
    """True when this text already states the agent cannot transact."""
    return bool(text) and bool(DISCLOSES.search(text) or DISCLOSURE in text)


def disclosure_due(
    *,
    message: str,
    tool_calls: list[dict] | None,
    already_disclosed: bool,
) -> bool:
    """Whether this turn owes the user the disclosure.

    Due when the conversation has not had it yet AND either the turn shows
    bookable options or the message asks outright to transact. `already_
    disclosed` is the caller's view of the conversation so far — this function
    holds no state of its own, so the same rule serves the agent (deciding) and
    the evaluator (checking).
    """
    if already_disclosed:
        return False
    return presents_bookable_options(tool_calls) or bool(TRANSACT.search(message or ""))


def ensure_disclosure(
    message: str,
    reply: str,
    *,
    tool_calls: list[dict] | None = None,
    already_disclosed: bool = False,
) -> str:
    """Return `reply`, prefixed with the disclosure if this turn owes one."""
    if discloses(reply):
        return reply
    if not disclosure_due(message=message, tool_calls=tool_calls,
                          already_disclosed=already_disclosed):
        return reply
    return f"{DISCLOSURE}\n\n{reply}" if reply else DISCLOSURE
