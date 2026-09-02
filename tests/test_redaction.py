"""Tests for span redaction.

Two obligations pull against each other and both have to hold: personal data
must not reach the observability backend, and the fields the evaluators read
must survive. `TraceConfig(hide_inputs=True)` satisfies the first by destroying
the second, which is why this redacts patterns rather than whole fields.
"""

from __future__ import annotations

import pytest

from common.redaction import RedactingSpanExporter, redact_attributes, redact_text


@pytest.mark.parametrize(
    ("raw", "marker"),
    [
        ("Book it for joaquin@example.com please.", "[REDACTED_EMAIL]"),
        ("Card 4111 1111 1111 1111 thanks.", "[REDACTED_CARD]"),
        ("Card 4111111111111111.", "[REDACTED_CARD]"),
        ("Call +1 415 555 0132.", "[REDACTED_PHONE]"),
        ("Call 415-555-0132.", "[REDACTED_PHONE]"),
        ("SSN 123-45-6789.", "[REDACTED_SSN]"),
        ("Passport L8912345 expires soon.", "[REDACTED_PASSPORT]"),
    ],
)
def test_personal_data_is_removed(raw: str, marker: str) -> None:
    out = redact_text(raw)
    assert marker in out
    assert not any(tok in out for tok in ("example.com", "4111", "555 0132",
                                          "555-0132", "123-45-6789", "L8912345"))


@pytest.mark.parametrize("text", [
    "Delta DL 883 departs 09:40 for $214.",
    "Hotel Lumiere $385/night - $1,540 for 4 nights.",
    "Flights on 2026-09-11, checkout 2026-06-14.",
    "Rated 4.7 out of 5.",
    "New York to Miami, then Miami to Tokyo.",
])
def test_the_data_the_evaluators_read_survives(text: str) -> None:
    """Redaction that ate prices, dates or flight numbers would silently break
    tool_result_exact, date_grounding and no_fabricated_flights at once."""
    assert redact_text(text) == text


def test_numeric_attributes_are_left_alone() -> None:
    attrs = {"llm.token_count.prompt": 1054, "llm.model_name": "claude-haiku-4-5",
             "input.value": "mail me at a@b.com"}
    out = redact_attributes(attrs)
    assert out["llm.token_count.prompt"] == 1054
    assert out["llm.model_name"] == "claude-haiku-4-5"
    assert "a@b.com" not in out["input.value"]


def test_an_unredactable_value_fails_closed() -> None:
    """A truncated trace is recoverable; exported personal data is not."""
    class Hostile(str):
        def __hash__(self):
            return 0

    exploding = {"input.value": Hostile("a@b.com")}
    # Force the failure path by making the regex sub raise.
    import common.redaction as r
    original = r.redact_value
    r.redact_value = lambda v: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        out = redact_attributes(exploding)
    finally:
        r.redact_value = original
    assert out["input.value"] == "[REDACTION_FAILED]"


def test_exporter_scrubs_spans_before_delegating() -> None:
    class Span:
        def __init__(self) -> None:
            self.name = "travel_agent"
            self._attributes = {"input.value": "reach me at a@b.com"}

    captured = {}

    class Inner:
        def export(self, spans):
            captured["value"] = spans[0]._attributes["input.value"]
            return "ok"

    span = Span()
    assert RedactingSpanExporter(Inner()).export([span]) == "ok"
    assert "a@b.com" not in captured["value"]
    assert "[REDACTED_EMAIL]" in captured["value"]


def test_the_canary_detects_a_broken_redaction_control() -> None:
    """`no_unredacted_pii` watches the control, not the agent.

    If redaction silently stops working — a regex edit, an exporter swap, a
    config change — nothing else in the loop notices, and the failure mode is
    believing you are covered when you are not.
    """
    from evals.agents.travel import evaluators as E  # noqa: F401  (registers)
    from evals.agents.travel.evaluators import no_unredacted_pii as ev

    clean = {"input": {"message": "I'm [REDACTED_EMAIL]."},
             "output": {"reply": "Delta DL 883 at 09:40 for $214.", "tool_calls": []}}
    assert ev.evaluate(clean)[0].score == 1.0

    leaked = {"input": {"message": "I'm joaquin@example.com, card 4111 1111 1111 1111."},
              "output": {"reply": "Booked.", "tool_calls": []}}
    assert ev.evaluate(leaked)[0].score == 0.0


def test_the_canary_does_not_re_leak_what_it_finds() -> None:
    """Its explanation is written back to Phoenix as an annotation. Quoting the
    match would copy the data into a second place — the exact failure it exists
    to report."""
    from evals.agents.travel import evaluators as E  # noqa: F401
    from evals.agents.travel.evaluators import no_unredacted_pii as ev

    leaked = {"input": {"message": "joaquin@example.com, card 4111 1111 1111 1111"},
              "output": {"reply": "ok", "tool_calls": []}}
    explanation = ev.evaluate(leaked)[0].explanation or ""
    assert "EMAIL" in explanation and "CARD" in explanation, "categories are reported"
    assert "joaquin@example.com" not in explanation
    assert "4111" not in explanation


def test_curation_redacts_before_it_persists() -> None:
    """Curation copies a turn into a dataset — a second store with its own
    lifecycle. If upstream redaction ever fails, this is the second line."""
    from evals.core.curate import Candidate, publish

    captured = {}

    class FakeClient:
        class datasets:
            @staticmethod
            def create_dataset(*, name, examples):
                captured["message"] = examples[0]["input"]["message"]
                return type("D", (), {"id": "ds-1"})()

    cand = Candidate(span_id="s1", message="Refund card 4111 1111 1111 1111 to a@b.com",
                     flagged_by=["x"], expected={}, needs_review=True, reason="r")

    from evals.core.config import AgentConfig

    conf = AgentConfig(agent="travel", project="p", dataset="d",
                       agent_span_name="s", curation={"candidates_dataset": "t"})
    publish([cand], conf, client=FakeClient())
    assert "4111" not in captured["message"]
    assert "a@b.com" not in captured["message"]
    assert "[REDACTED_CARD]" in captured["message"]


def test_exactly_one_processor_carries_every_span_out() -> None:
    """The redaction control depends on Phoenix replacing its default processor.

    Stock OTel `add_span_processor` appends, which would leave the default OTLP
    exporter running alongside ours and every span exported twice — once
    unredacted. Phoenix's TracerProvider overrides it to replace. That is a
    property of a dependency, not of our code, so it is pinned here: if a
    Phoenix upgrade changes it, redaction silently stops covering the export
    path and this test is what says so.
    """
    import os

    import pytest

    if not os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        pytest.skip("needs a collector endpoint to build a provider")

    from backend.tracing import configure_tracing
    from common.redaction import RedactingSpanExporter

    provider = configure_tracing()
    processors = provider._active_span_processor._span_processors
    assert len(processors) == 1, "a second processor would export unredacted spans"
    assert isinstance(processors[0].span_exporter, RedactingSpanExporter)
