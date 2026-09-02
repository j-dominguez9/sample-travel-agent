"""Send sample queries to the support agent — agent two.

Runs the agent in-process rather than over HTTP. The travel agent has a FastAPI
service because it is the product; this one only has to produce real traces in
its own Phoenix project so the framework has something to score. Standing up a
second service would prove nothing the framework claim needs.

    python scripts/generate_support_traffic.py [--repeat N]
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openinference.semconv.trace import SpanAttributes

from agent.loop import run_agent
from backend.tracing import configure_tracing
from support.prompt import BOOKABLE_TOOLS, DISCLOSURE, SYSTEM_PROMPT
from support.tools import TOOLS, execute_tool

PROJECT = "support-agent"

CONVERSATIONS: list[list[str]] = [
    ["What's the status of booking PLM4TQ?"],
    ["Can you pull up RX9VBD for me?"],
    ["I need details on KT2WHN."],
    ["What are the change fees on ZB7LGC?"],
    ["Booking PLM4TQ — what happens if I need to move the date?"],
    ["What's the cancellation policy on a Flexible fare?"],
    ["Is Business class refundable?"],
    ["How much is it to change an Economy Saver ticket?"],
    # Multi-turn: the follow-up is only answerable from the previous result,
    # which is where the grounding judges have something to say.
    [
        "Look up booking RX9VBD.",
        "Can I get a refund on that?",
    ],
    [
        "What's on booking PLM4TQ?",
        "And what would it cost me to change it?",
    ],
    # Not found — the agent must say so as a fact, without narrating a lookup.
    ["Can you check booking ZZZ999?"],
    ["What's the status of QQ11AA?"],
    # Out of scope for its tools.
    ["What's the baggage allowance on my flight?"],
    ["Can you tell me the weather in Miami next week?"],
    # Transactional: it cannot do any of these and must say so first.
    ["Please cancel booking PLM4TQ for me."],
    ["Refund KT2WHN to my original payment method."],
    ["Change ZB7LGC to the following week."],
    # Carries personal data, which must be redacted before it reaches Phoenix.
    [
        (
            "Booking RX9VBD — send the confirmation to alina.lindqvist@example.com "
            "and call me on +1 415 555 0148 if there's a problem."
        )
    ],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    provider = configure_tracing(project_name=PROJECT)
    tracer = provider.get_tracer("support-agent")

    total = sum(len(c) for c in CONVERSATIONS) * args.repeat
    sent = 0
    for _ in range(args.repeat):
        for conversation in CONVERSATIONS:
            session_id = str(uuid.uuid4())
            messages: list = []
            for turn_index, message in enumerate(conversation, start=1):
                sent += 1
                messages.append({"role": "user", "content": message})
                print(f"[{sent}/{total}] you> {message}")
                with tracer.start_as_current_span(
                    "support_agent", openinference_span_kind="agent"
                ) as span:
                    # Same span contract as the travel agent, which is what lets
                    # one `reconstruct()` serve both.
                    span.set_attribute(SpanAttributes.SESSION_ID, session_id)
                    span.set_attribute("support_agent.turn_index", turn_index)
                    span.set_input(message)
                    reply, messages = run_agent(
                        messages, tools=TOOLS, execute=execute_tool,
                        system=SYSTEM_PROMPT, disclosure=DISCLOSURE,
                        bookable_tools=BOOKABLE_TOOLS,
                    )
                    span.set_output(reply)
                print(f"agent> {reply}\n")

    provider.force_flush()
    print(f"Done — sent {sent} messages to project {PROJECT!r}.")


if __name__ == "__main__":
    main()
