"""Strip personal data from spans before they leave the process.

The customer's requirement was "PII redacted in observability". The obvious
implementation is OpenInference's `TraceConfig(hide_inputs=True)`, and it is the
wrong one here: it blanks `input.value` and `output.value` entirely, and those
two attributes are what the entire evaluation system reads. Hiding them would
leave traces that satisfy the requirement and an eval framework with nothing to
evaluate — the monitor could not reconstruct a turn, the judges would have no
reply to grade, and curation would produce empty candidates.

So this redacts *patterns* rather than *fields*. An email address is replaced;
the sentence around it survives, and `flight_direction` still sees which flights
came back.

Two placement decisions:

**At export, not at the call site.** Most spans here are created by the Anthropic
instrumentor, not by our code — we never touch `llm.input_messages`. Wrapping the
exporter is the one point every span passes through regardless of who made it.

**Fail closed.** If redaction raises on a value, that value is replaced wholesale
rather than passed through. A truncated trace is recoverable; exported PII is not.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)

#: Ordered: card numbers are matched before bare digit runs so a card is never
#: partially consumed by a looser pattern first.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    # 13-19 digits with optional separators — the ISO card-number range. Prices
    # and flight numbers in this domain are far shorter, so the floor of 13
    # keeps "$1,540" and "DL 883" out of it.
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Passport-ish: one or two letters then 6-9 digits.
    ("PASSPORT", re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")),
    # E.164 and common separated forms; requires a + or 10+ digits so a date
    # like 2026-09-01 cannot match.
    ("PHONE", re.compile(r"\+\d[\d ().-]{8,}\d|\b\d{3}[ .-]\d{3}[ .-]\d{4}\b")),
)

#: Attribute keys never worth scanning — numeric counts, model names, kinds.
#: Skipping them is a cost decision, not a safety one: they cannot hold prose.
#: `session.id` is the exception that is about correctness: it is a UUID we
#: generate, and a numeric-looking run inside one can trip the CARD pattern.
#: Redacting it would silently break session grouping in Phoenix.
_SKIP_SUFFIXES = (".token_count", "_count", ".kind", ".model_name", ".provider",
                  ".system", "session.id", ".turn_index")


def redact_text(value: str) -> str:
    for label, pattern in PATTERNS:
        value = pattern.sub(f"[REDACTED_{label}]", value)
    return value


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (list, tuple)):
        return type(value)(redact_value(v) for v in value)
    return value


def redact_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in attributes.items():
        if any(key.endswith(s) for s in _SKIP_SUFFIXES):
            out[key] = value
            continue
        try:
            out[key] = redact_value(value)
        except Exception:  # noqa: BLE001
            # Fail closed: an unredactable value is dropped, not exported raw.
            logger.warning("redaction failed; dropping value", extra={"attribute": key})
            out[key] = "[REDACTION_FAILED]"
    return out


class RedactingSpanExporter(SpanExporter):
    """Wraps an exporter and scrubs every span's attributes on the way out."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            attributes = getattr(span, "_attributes", None)
            if not attributes:
                continue
            try:
                span._attributes = redact_attributes(dict(attributes))
            except Exception:  # noqa: BLE001
                logger.warning("span redaction failed; exporting no attributes",
                               extra={"span": span.name})
                span._attributes = {}
        return self._inner.export(spans)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._inner.shutdown()
