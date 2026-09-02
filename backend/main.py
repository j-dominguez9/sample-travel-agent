import logging
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from openinference.semconv.trace import SpanAttributes
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel

from agent.loop import run_agent
from agent.prompt import PROMPT_VERSION
from backend.tracing import configure_tracing
from common.logging import configure_logging

load_dotenv()
configure_logging()
tracer_provider = configure_tracing()
tracer = tracer_provider.get_tracer("travel-agent")
logger = logging.getLogger(__name__)


app = FastAPI(title="Travel Agent")
FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)


CONVERSATIONS: dict[str, list] = {}


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conversation_id = req.conversation_id or str(uuid.uuid4())
    messages = CONVERSATIONS.get(conversation_id, [])
    messages.append({"role": "user", "content": req.message})
    # 1-based position of this message in its conversation. Without it a turn
    # has no conversation identity at all, and an evaluator cannot tell an
    # opening out-of-scope request ("do I need a visa?") from a follow-up
    # answered out of context ("which of those is cheapest?") — they look
    # identical, because neither calls a tool.
    #
    # Only string content counts. The Anthropic tool-calling loop feeds results
    # back as *user*-role messages carrying a list of tool_result blocks
    # (`loop.py`), so counting every user role makes one search look like two
    # turns — the second message of a two-turn conversation reported turn 3.
    turn_index = sum(
        1 for m in messages
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    )
    with tracer.start_as_current_span(
        "travel_agent", openinference_span_kind="agent"
    ) as span:
        # `session.id` is the OpenInference convention Phoenix groups sessions
        # by, so setting it lights up session views in the UI as well as
        # feeding the evaluators.
        span.set_attribute(SpanAttributes.SESSION_ID, conversation_id)
        span.set_attribute("travel_agent.turn_index", turn_index)
        # Which prompt produced this turn. Without it a prompt A/B is only as
        # good as your ability to remember when you restarted the server —
        # comparing v5 to v6 by timestamp silently pooled the two, because the
        # runs were minutes apart and the window covered both.
        span.set_attribute("travel_agent.prompt_version", PROMPT_VERSION)
        span.set_input(req.message)
        reply, messages = run_agent(messages)
        span.set_output(reply)
    CONVERSATIONS[conversation_id] = messages
    return ChatResponse(reply=reply, conversation_id=conversation_id)
