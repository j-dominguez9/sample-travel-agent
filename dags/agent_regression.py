"""Nightly regression run against the golden dataset, one DAG per agent.

This is the job CI structurally cannot do. CI fires on a commit, so it only ever
answers "did this change break something?". An agent can regress with no commit
at all — a provider ships a new model revision, a prompt is edited in another
service, fixture data shifts underneath it. Running the same dataset every night
and comparing to the previous run is what surfaces that.

It also runs the LLM judges, which the PR gate deliberately does not: judge
output varies between runs, so it belongs on a trend rather than in a blocking
check. `repetitions` comes from config for the same reason — a single run of a
non-deterministic task is not a measurement.

    run_experiment ── compare_to_previous ── alert_on_drift
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
        dag_id=f"agent_regression_{conf.agent}",
        schedule=conf.regression.get("schedule", "0 3 * * *"),
        start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
        catchup=False,
        max_active_runs=1,
        tags=["evals", "regression", conf.agent],
        default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=10)},
        doc_md=__doc__,
    )
    def regression() -> None:
        @task
        def run_experiment(ds: str | None = None) -> dict:
            # `ds` is a reserved Airflow context key: the scheduler injects the
            # logical date when the parameter defaults to None. Giving it a
            # template string as a default is rejected at parse time.
            from datetime import UTC, datetime

            ds = ds or datetime.now(UTC).date().isoformat()

            import agent.loop as agent_loop
            from evals.agents.travel import (  # noqa: F401  (registers)
                evaluators,
                judges,
            )
            from evals.core import runner

            name = f"nightly-{conf.agent}-{ds}"
            runner.run(
                agent=conf.agent,
                dataset_name=conf.dataset,
                agent_fn=agent_loop.run_agent,
                experiment_name=name,
                names=conf.regression.get("evaluators"),
                repetitions=int(conf.regression.get("repetitions", 1)),
            )
            return {"experiment": name}

        @task
        def compare_to_previous(current: dict) -> dict:
            """Diff tonight's rates against the most recent prior nightly."""
            from phoenix.client import Client

            from evals.report import _g, load_by_id, resolve

            client = Client()
            dataset = client.datasets.get_dataset(dataset=conf.dataset)
            nightlies = [
                e for e in client.experiments.list(dataset_id=dataset.id)
                if str(_g(e, "name", "")).startswith(f"nightly-{conf.agent}-")
            ]
            if len(nightlies) < 2:
                return {"status": "no baseline yet", "experiment": current["experiment"]}

            # Experiment names are `nightly-{agent}-{ds}` and are NOT unique: a
            # task retry or a same-day re-trigger creates a second experiment
            # with the identical name. Picking the second-newest *name* then
            # returns tonight's own name, `resolve` maps it back to tonight's
            # run, and the DAG compares the run to itself — drift always empty,
            # "no metric moved" forever. Select by id, excluding this run's.
            current_id = resolve(client, dataset.id, current["experiment"])
            older = [e for e in nightlies if str(_g(e, "id")) != current_id]
            if not older:
                return {"status": "no baseline yet", "experiment": current["experiment"]}
            previous_experiment = max(older, key=lambda e: str(_g(e, "id")))
            previous = str(_g(previous_experiment, "name"))
            previous_id = str(_g(previous_experiment, "id"))
            now_scores, _ = load_by_id(client, current_id)
            was_scores, _ = load_by_id(client, previous_id)

            def rates(scores: dict) -> dict[str, float]:
                per: dict[str, list[float]] = {}
                for (_run, name), value in scores.items():
                    per.setdefault(name, []).append(value)
                return {k: sum(v) / len(v) for k, v in per.items() if v}

            now, was = rates(now_scores), rates(was_scores)
            tolerance = float(conf.regression.get("drift_tolerance", 0.10))
            drift = {
                k: {"previous": was[k], "current": now[k], "delta": now[k] - was[k]}
                for k in sorted(set(now) & set(was))
                if abs(now[k] - was[k]) > tolerance
            }
            # A drift number says something moved; the digest says which turns
            # moved it. Built only when there is drift, since it costs a call.
            digest_text = ""
            if drift:
                from evals.agents.travel import evaluators, judges  # noqa: F401
                from evals.core import diagnose as dx

                records, scores = dx.from_experiment(
                    client, dataset.id, current["experiment"]
                )
                digest_text, _ = dx.digest(records, scores, conf)

            return {
                "status": "compared",
                "baseline": previous,
                "experiment": current["experiment"],
                "drift": drift,
                "tolerance": tolerance,
                "digest": digest_text,
            }

        @task
        def alert_on_drift(comparison: dict) -> None:
            if comparison["status"] != "compared":
                print(f"[{conf.agent}] {comparison['status']} — nothing to compare")
                return
            drift = comparison["drift"]
            if not drift:
                print(
                    f"[{conf.agent}] no metric moved more than "
                    f"{comparison['tolerance']:.0%} against {comparison['baseline']}"
                )
                return
            owners = ", ".join(conf.alerting.get("owners", []))
            print(f"[{conf.agent}] DRIFT -> {owners}  (vs {comparison['baseline']})")
            for name, d in drift.items():
                print(
                    f"  {name}: {d['previous']:.1%} -> {d['current']:.1%} "
                    f"({d['delta']:+.1%})"
                )
            # A drift number says something moved; the digest says which turns
            # moved it and offers somewhere to start.
            print()
            print(comparison.get("digest") or "(no digest available)")

        alert_on_drift(compare_to_previous(run_experiment()))

    return regression()


for _conf in agent_config.discover():
    globals()[f"agent_regression_{_conf.agent}"] = build(_conf)
