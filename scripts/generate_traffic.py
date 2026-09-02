"""Send a batch of sample user queries to the travel agent API.

Usage:
    python scripts/generate_traffic.py [base_url] [--repeat N] [--seed N]

The API server must be running first:
    uv run fastapi dev backend/main.py

This demo has no live traffic, so this script *is* the production stream the
monitoring DAG reads. That makes its shape load-bearing rather than incidental:
the judges only grade the turns they apply to, and `min_sample` suppresses any
evaluator under 20 of them. A run that is all happy-path flight searches leaves
`graceful_alternative` at n=3 and silently unreported. The mix below is
therefore weighted to keep every evaluator above its floor — roughly a third of
conversations end in nothing bookable or nothing in scope, which is high for a
real travel agent but is what makes the thin evaluators visible.

One pass is ~55 turns. A three-hour monitoring window wants a couple of passes;
`--repeat 2` shuffles between them so the ordering differs.
"""

import argparse
import os
import random
import time

import httpx

# Each entry is a conversation: a list of one or more user messages sent in order.
CONVERSATIONS = [
    ["Find me a flight from New York to Miami on March 12, 2026."],
    ["What flights are there from San Francisco to Tokyo on April 20, 2026?"],
    ["I need a hotel in Paris from June 10 to June 14, 2026."],
    ["Can you find hotels in Chicago for May 5 to May 8, 2026?"],
    ["What's the weather like in Miami on July 15, 2026?"],
    ["How's the weather looking in Tokyo on April 22, 2026?"],
    ["Plan a 3-day trip to Chicago for me."],
    [
        "I'm thinking about a weekend in Miami in early August. Any flights from New York on August 7, 2026?",
        "Great, can you add a hotel for that weekend too?",
    ],
    ["Show me flights from London to Paris on September 3, 2026."],
    ["I need to get from Tokyo to Los Angeles on May 2, 2026 — what flights are there?"],
    ["Put together a 5-day itinerary for Paris, arriving June 10, 2026."],
    ["I want to fly from Chicago to Denver on October 2, 2026 — what are my options?"],
    ["Find me a hotel in New York for the nights of March 20 to 23, 2026."],
    ["I need a flight from Miami to Tokyo next Friday."],
    ["Can you get me a hotel in Denver for this weekend?"],
    ["What hotels are available in Paris from April 3 to April 7, 2026?"],
    ["Are there any flights from Denver to Miami on August 14, 2026?"],
    ["Do I need a visa to visit Japan as a US citizen?"],
    ["I booked a flight through you last month and need a refund — can you process that?"],
    ["How much would a hotel in London cost per night in euros?"],
    ["What's the weather going to be in London next Tuesday?"],
    ["Find me a hotel in Austin for South by Southwest."],
    # Multi-turn: the follow-up can only be answered from the previous tool
    # result, so these are where the grounding judges have the most to say.
    [
        "Find flights from Denver to Chicago on July 4, 2026.",
        "Which of those is cheapest?",
        "Book me a hotel there for two nights after that.",
    ],
    [
        "I need a hotel in Tokyo from May 1 to May 4, 2026.",
        "What's the weather like while I'm there?",
    ],
    [
        "Show me flights from Miami to New York on June 2, 2026.",
        "Anything earlier in the day?",
    ],
    ["Plan a 4-day trip to Tokyo in April 2026."],
    ["Build me a 2-day Miami itinerary for August 2026."],
    ["What's the weather in Chicago on May 6, 2026?"],
    ["Flights from Paris to London on July 19, 2026, please."],
    ["I need a hotel in Denver from September 14 to September 17, 2026."],
    ["Are there flights from Los Angeles to Tokyo on June 25, 2026?"],
    # Past dates. The agent is instructed (prompt v4) to search these rather
    # than refuse, and the judges now carry today's date so they can check it.
    ["Did any flights run from New York to Miami on January 5, 2026?"],
    ["What hotels were available in Chicago from February 2 to February 5, 2026?"],
    # Nothing available / out of scope — these feed graceful_alternative, the
    # thinnest evaluator in the suite.
    ["Find me a flight from New York to Reykjavik on March 5, 2026."],
    ["Any hotels in Lagos for November 2026?"],
    ["What's the weather in Nairobi on December 1, 2026?"],
    ["I need a flight from Boston to Seoul on July 8, 2026."],
    ["Can you recommend a good travel insurance policy?"],
    ["What's the checked baggage allowance on Delta?"],
    ["Can you change the name on my existing reservation?"],
    ["Is it safe to travel to Cairo right now?"],
    ["What's the best credit card for earning airline miles?"],
    ["Book me the cheapest flight to anywhere warm and charge my card on file."],
    # Realistic requests that carry personal data. Travellers volunteer contact
    # and payment details unprompted, so these exercise the redaction path in
    # traffic rather than only in unit tests. The model still receives the full
    # message — redaction governs what reaches observability, not what the agent
    # can read.
    [
        (
            "Find me a flight from New York to Miami on October 1, 2026. "
            "You can send the confirmation to joaquin.dominguez@example.com."
        )
    ],
    [
        (
            "I need a hotel in Paris from October 5 to October 9, 2026. "
            "Call me on +1 415 555 0132 if anything changes."
        )
    ],
    [
        (
            "Hold the New York to Chicago flight on October 8, 2026 — "
            "put the deposit on 4111 1111 1111 1111."
        )
    ],
    [
        "My passport is L8912345. What is the weather in Tokyo on October 20, 2026?"
    ],
    [
        (
            "Book the Chicago to Denver flight on November 3, 2026 under "
            "marcus.webb@example.org — reach me at 415-555-0177 if it's full."
        )
    ],
    [
        (
            "I'm checking into the Paris hotel on December 2, 2026. "
            "They asked for my SSN, 402-11-9863 — do they normally need that?"
        )
    ],
    [
        (
            "Find flights from Tokyo to Los Angeles on November 12, 2026. "
            "My frequent flyer is AA4471829 and my card is 5500 0000 0000 0004."
        )
    ],
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base_url", nargs="?",
                    default=os.getenv("TRAVEL_AGENT_URL", "http://localhost:8000"))
    ap.add_argument("--repeat", type=int, default=1,
                    help="passes over the conversation set (default 1, ~55 turns)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the shuffle so a run is reproducible")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total = sum(len(c) for c in CONVERSATIONS) * args.repeat
    sent = 0

    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        client.get("/health").raise_for_status()

        for pass_no in range(args.repeat):
            batch = list(CONVERSATIONS)
            if pass_no:
                # Vary the ordering between passes so repeated runs do not lay
                # down an identical trace timeline.
                rng.shuffle(batch)
            for conversation in batch:
                conversation_id = None
                for message in conversation:
                    sent += 1
                    print(f"[{sent}/{total}] you> {message}")
                    resp = client.post(
                        "/chat",
                        json={"message": message, "conversation_id": conversation_id},
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    conversation_id = body["conversation_id"]
                    print(f"agent> {body['reply']}\n")
                    time.sleep(0.5)

    print(f"Done — sent {sent} messages across "
          f"{len(CONVERSATIONS) * args.repeat} conversations.")


if __name__ == "__main__":
    main()
