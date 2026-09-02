"""Trace the evaluation system's own model calls.

The cost report asks two questions: what does the agent cost, and what does
watching it cost. Phoenix answered the first natively and the second as `$0.00`,
which was wrong rather than free — the judges' Anthropic calls happen inside an
Airflow worker with no tracer configured, so nothing recorded them. Phoenix's
`evaluators` project does hold spans, but those are evaluation *results*: they
carry a verdict and no token counts, so they price at zero.

Turning this on puts judge calls in their own project with real token counts,
which Phoenix then prices from the same table it uses for the agent. That makes
"the eval stack costs more than the agent it watches" a measured number rather
than an assertion, which is the number the customer needs before scaling this
to 1M conversations a year.

Kept separate from `backend/tracing.py` on purpose. That module traces the
product; this one traces the tooling, into a different project, so the agent's
own cost is never inflated by the cost of measuring it.
"""

from __future__ import annotations

import logging
import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from common.redaction import RedactingSpanExporter

logger = logging.getLogger(__name__)

_ENABLED: set[str] = set()


def enable(project_name: str) -> bool:
    """Route this process's LLM calls to `project_name`. Idempotent.

    Returns True when tracing is active. Never raises: a monitoring sweep must
    not fail because its own cost accounting could not be set up — losing a
    cost datapoint is recoverable, losing the sweep is not.
    """
    if project_name in _ENABLED:
        return True

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        logger.info("eval tracing skipped: no PHOENIX_COLLECTOR_ENDPOINT")
        return False

    try:
        from phoenix.otel import register

        provider = register(
            project_name=project_name,
            auto_instrument=True,
            batch=True,
            # This runs inside a worker that already talks to Phoenix as a
            # client. Claiming the global provider would put our own tracing in
            # the way of anything else in the process.
            set_global_tracer_provider=False,
        )
        # Same redaction as the product path: a judge prompt contains the user's
        # message, so these spans can carry PII exactly like the agent's can.
        provider.add_span_processor(
            BatchSpanProcessor(
                RedactingSpanExporter(
                    OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
                )
            )
        )
    except Exception:
        logger.warning("eval tracing could not be enabled", exc_info=True)
        return False

    _ENABLED.add(project_name)
    logger.info("eval tracing enabled", extra={"project": project_name})
    return True
