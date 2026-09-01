"""Hourly production monitoring, one DAG per configured agent.

The DAG is deliberately thin: every task is a few lines calling `evals.core`,
which is importable and tested without Airflow. Orchestration logic that only
runs inside a scheduler is logic nobody can test.

Task boundaries follow cost, not tidiness. `sweep` is the only task that spends
money — it calls the LLM judges — and it writes its results to Phoenix as span
annotations. Everything downstream re-reads those annotations rather than
receiving them through XCom. That keeps each task independently re-runnable
(retry `check_thresholds` without paying for the judges again) and avoids
pushing DataFrames through a metadata database that isn't built for them.

    sweep ─┬─ check_thresholds ── alert
           └─ curate_candidates
"""

from __future__ import annotations

import pendulum

try:  # Airflow 3.x
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

from evals.core import config as agent_config


def build(conf) -> object:
    @dag(
        dag_id=f"agent_monitoring_{conf.agent}",
        schedule=conf.schedule,
        start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
        catchup=False,
        max_active_runs=1,
        tags=["evals", "monitoring", conf.agent],
        default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=5)},
        doc_md=__doc__,
    )
    def monitoring() -> None:
        @task
        def sweep() -> dict:
            """Sample recent traffic, score it, write annotations to Phoenix.

            The only task that calls a model. Its durable output is the
            annotations, not the return value.
            """
            from evals.agents.travel import (  # noqa: F401  (registers)
                evaluators,
                judges,
            )
            from evals.core import monitor

            results = monitor.sweep(
                agent=conf.agent,
                project=conf.project,
                since_minutes=int(conf.monitoring.get("window_minutes", 60)),
                limit=int(conf.monitoring.get("limit", 200)),
                agent_span_name=conf.agent_span_name,
            )
            return {"annotations": len(results),
                    "spans": int(results["span_id"].nunique()) if len(results) else 0}

        @task
        def check_thresholds(swept: dict) -> dict:
            """Compare rates against config. Reads annotations back, never rescores."""
            from evals.agents.travel import evaluators, judges  # noqa: F401
            from evals.core import monitor, thresholds

            scores = monitor.read_annotations(
                project=conf.project,
                agent_span_name=conf.agent_span_name,
                since_minutes=int(conf.monitoring.get("window_minutes", 60)),
                limit=int(conf.monitoring.get("limit", 200)),
            )
            report = thresholds.evaluate(scores, conf)
            print(report.summary())
            return {
                "should_page": report.should_page,
                "breaches": [b.describe() for b in report.breaches],
                "measured": {k: [v[0], v[1]] for k, v in report.measured.items()},
                "swept": swept,
            }

        @task
        def diagnose(swept: dict) -> str:
            """Build the digest the alert recipient actually needs.

            Runs on any failing turns, not only on a threshold breach — the
            relative-date failures that mattered most were suppressed by the
            thin-sample guard at n=9, and a breach-gated analysis would have
            said nothing about them.
            """
            from datetime import UTC, datetime, timedelta

            from evals.agents.travel import evaluators, judges  # noqa: F401
            from evals.core import diagnose as dx
            from evals.core import monitor

            window = int(conf.monitoring.get("window_minutes", 60))
            limit = int(conf.monitoring.get("limit", 200))
            # Same window as the annotations, or the two sets need not overlap:
            # an unwindowed limit=200 on a busy project can miss the very spans
            # the scores refer to, leaving clusters with no turns attached.
            start = datetime.now(UTC) - timedelta(minutes=window)
            spans = monitor.Client().spans.get_spans_dataframe(
                project_identifier=conf.project, start_time=start, limit=limit
            )
            records = monitor.reconstruct(spans, agent_span_name=conf.agent_span_name)
            scores = monitor.read_annotations(
                project=conf.project,
                agent_span_name=conf.agent_span_name,
                since_minutes=window,
                limit=limit,
            )
            text, _ = dx.digest(records, scores, conf)
            print(text)
            return text

        @task
        def alert(report: dict, digest: str) -> None:
            """Escalate only the classes the customer named as page-worthy.

            Deliberately a print: routing to PagerDuty, Slack or email is a
            per-customer integration, and inventing one here would be guessing.
            The decision of *what* is page-worthy is the part that belongs in
            this repo, and it lives in the agent's config.
            """
            if not report["breaches"]:
                print(f"[{conf.agent}] all measured evaluators within threshold")
                return
            owners = ", ".join(conf.alerting.get("owners", []))
            level = "PAGE" if report["should_page"] else "NOTIFY"
            print(f"[{conf.agent}] {level} -> {owners}")
            for line in report["breaches"]:
                print(f"  {line}")
            print()
            print(digest)

        @task
        def curate_candidates(swept: dict) -> dict:
            """Flagged production turns become candidate golden cases.

            The half that makes this a feedback loop: the golden set can only
            contain failures somebody wrote down, and production keeps producing
            ones nobody did. Candidates land in their own dataset for review and
            are never merged automatically.
            """
            from datetime import UTC, datetime, timedelta

            from evals.agents.travel import evaluators, judges, truth  # noqa: F401
            from evals.core import curate, monitor

            if not conf.curation.get("enabled", False):
                return {"candidates": 0, "skipped": "curation disabled in config"}

            window = int(conf.monitoring.get("window_minutes", 60))
            start = datetime.now(UTC) - timedelta(minutes=window)
            spans = monitor.Client().spans.get_spans_dataframe(
                project_identifier=conf.project,
                start_time=start,
                limit=int(conf.monitoring.get("limit", 200)),
            )
            records = monitor.reconstruct(spans, agent_span_name=conf.agent_span_name)
            scores = monitor.read_annotations(
                project=conf.project,
                agent_span_name=conf.agent_span_name,
                since_minutes=int(conf.monitoring.get("window_minutes", 60)),
                limit=int(conf.monitoring.get("limit", 200)),
            )
            candidates = curate.collect(records, scores, conf, truth)
            print(curate.summary(candidates))
            dataset_id = curate.publish(candidates, conf)
            return {
                "candidates": len(candidates),
                "needs_review": sum(1 for c in candidates if c.needs_review),
                "dataset_id": dataset_id,
            }

        swept = sweep()
        alert(check_thresholds(swept), diagnose(swept))
        curate_candidates(swept)

    return monitoring()


# One DAG per agent, discovered from evals/agents/*/config.yaml.
for _conf in agent_config.discover():
    globals()[f"agent_monitoring_{_conf.agent}"] = build(_conf)
