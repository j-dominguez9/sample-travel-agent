import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

import anthropic
from opentelemetry import trace

from agent.config import MAX_TOKENS, MODEL
from agent.prompt import system_prompt
from agent.tools import TOOLS, execute_tool
from common.capabilities import discloses, ensure_disclosure

client = anthropic.Anthropic()


@contextmanager
def _tool_span(name: str, tool_input: Any) -> Iterator[Any]:
    """Span for one tool call, with the OpenInference extras when available.

    Tracing must stay optional. Phoenix's `register()` installs a provider whose
    tracer accepts `openinference_span_kind` and yields spans with set_input and
    set_output; with no provider installed — the CLI, a unit test, anyone who
    hasn't configured Phoenix — the same call resolves to a NoOpTracer that
    rejects those and raises. The agent must not depend on its observability
    stack being wired up to answer a question, so this degrades to a plain span
    instead of failing.
    """
    tracer = trace.get_tracer("travel-agent")
    with ExitStack() as stack:
        try:
            # ProxyTracer is lazy: it builds the context manager without
            # complaint and only delegates on __enter__, so the unsupported
            # kwarg surfaces there rather than at the call. Entering through
            # ExitStack is what puts the failure somewhere catchable.
            span = stack.enter_context(
                tracer.start_as_current_span(name, openinference_span_kind="tool")
            )
        except TypeError:
            span = stack.enter_context(tracer.start_as_current_span(name))

        _record(span, "set_input", tool_input)
        yield span


def _record(span: Any, method: str, value: Any) -> None:
    """Attach an OpenInference attribute when the span can carry one.

    Two independent things have to hold, and checking only the first is how
    this broke: the tracer must accept OpenInference kwargs, AND the span it
    returns must be recording. With a provider installed but no exporter — a
    pytest run with tracing switched off — the tracer is a real OITracer that
    happily accepts `openinference_span_kind`, yet hands back an
    OpenInferenceSpan wrapping a NonRecordingSpan. That object *has*
    `set_input`; calling it raises "Cannot set input attributes on a
    non-OpenInference span", because the kind attribute was never stored.

    A non-recording span discards attributes anyway, so skipping is free.
    """
    if not getattr(span, "is_recording", lambda: False)():
        return
    setter = getattr(span, method, None)
    if setter is None:
        return
    try:
        setter(value)
    except ValueError:
        # The span is recording but not OpenInference-flavoured. Tracing is a
        # side effect; it must never take down the turn it is observing.
        pass


def _record_output(span: Any, result: Any) -> None:
    _record(span, "set_output", result)


def _assistant_texts(messages: list) -> Iterator[str]:
    """Text the assistant has already said in this conversation.

    Content is either a plain string (a turn this guardrail rewrote) or the
    SDK's list of content blocks, so both shapes have to be handled.
    """
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, str):
            yield content
            continue
        for block in content or []:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                yield text


def run_agent(
    messages: list,
    *,
    tools: list | None = None,
    execute: Any = None,
    system: str | None = None,
    disclosure: str | None = None,
    bookable_tools: frozenset[str] | None = None,
) -> tuple[str, list]:
    """Run one user turn through the tool-calling loop.

    `messages` must end with the latest user message. Returns the assistant's
    reply text and the updated message history.

    The keyword arguments exist so a second agent can reuse this loop with its
    own tools and prompt. They default to the travel agent, so every existing
    caller is unaffected — the loop itself was never travel-specific, only its
    imports were.
    """
    tools = TOOLS if tools is None else tools
    execute = execute_tool if execute is None else execute
    # Whether the conversation has already been told, read from history before
    # this turn adds to it.
    already_disclosed = any(discloses(t) for t in _assistant_texts(messages))
    # The user's own message: the last user turn whose content is a string. The
    # tool results appended below also carry the "user" role, with a list.
    last_user = next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        "",
    )
    calls: list[dict] = []

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system or system_prompt(),
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                with _tool_span(block.name, block.input) as span:
                    result = execute(block.name, block.input)
                    _record_output(span, result)
                calls.append({"name": block.name, "output": result})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    reply = "".join(block.text for block in response.content if block.type == "text")

    # Guardrail, not a guideline: two prompt versions asked the model to say it
    # cannot transact and neither beat doing nothing (see `capabilities.py`).
    # The trigger is "has this conversation seen bookable options", not "did the
    # user use a booking word" — the sentence that puts a user at risk is
    # usually "I'll take it", which contains no such word.
    disclosed = ensure_disclosure(
        last_user, reply, tool_calls=calls, already_disclosed=already_disclosed,
        disclosure=disclosure, bookable_tools=bookable_tools,
    )
    if disclosed != reply:
        # Keep history equal to what the user actually saw. Otherwise the model
        # never learns the disclosure was made, repeats it on the next turn, and
        # `already_disclosed` reads False forever.
        messages[-1] = {"role": "assistant", "content": disclosed}
    return disclosed, messages
