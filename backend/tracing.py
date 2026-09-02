import logging
import os

from openinference.instrumentation import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import register

from common.redaction import RedactingSpanExporter

logger = logging.getLogger(__name__)


def configure_tracing(project_name: str = "travel-agent") -> TracerProvider:
    """Otel for phoenix observability platform.

    `project_name` is a parameter because agent two traces to its own project.
    Keeping tenants in separate Phoenix projects is what lets cost, thresholds
    and sweeps be per-agent without any of them filtering the others out.
    """

    if not os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        raise KeyError("PHOENIX_COLLECTOR_ENDPOINT envvar not set.")
    tracer_provider = register(
        project_name=project_name, auto_instrument=True, batch=True
    )

    # Every span leaves through here, including the ones the Anthropic
    # instrumentor creates and we never touch. Phoenix's TracerProvider
    # overrides `add_span_processor` to *replace* its default rather than append
    # to it — stock OTel would append — so nothing gets a second, unredacted
    # path out. `register()` says as much on startup: "Using a default
    # SpanProcessor. `add_span_processor` will overwrite this default."
    #
    # That single-processor guarantee is load-bearing, and `test_redaction.py`
    # asserts it. RedactingSpanExporter scrubs `span._attributes` in place, so a
    # second processor added here would share the same ReadableSpan and see
    # redacted or raw attributes depending on which ran first. If you ever need
    # a second exporter, give it its own RedactingSpanExporter — do not assume
    # this one covers it.
    endpoint = os.environ["PHOENIX_COLLECTOR_ENDPOINT"].rstrip("/")
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            RedactingSpanExporter(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        )
    )

    logger.info(
        "tracing enabled", extra={"collector": os.environ["PHOENIX_COLLECTOR_ENDPOINT"]}
    )
    return tracer_provider
