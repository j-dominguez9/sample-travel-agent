"""Run the golden-dataset experiment against a chosen checkout of the agent.

    python -m evals.run_experiment --name current-fixed
    python -m evals.run_experiment --name baseline-main --agent-root /path/to/main

`--agent-root` puts a different checkout of the `agent` package first on the
import path, so the same evaluators grade a different implementation. The
evaluators and `truth.py` always resolve from *this* tree, anchored to this
tree's fixtures — the reference answer must not move when the agent does.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="experiment name in Phoenix")
    ap.add_argument("--agent-root", default=None,
                    help="checkout whose `agent` package to test (default: this tree)")
    ap.add_argument("--dataset", default="travel-golden-v1")
    ap.add_argument("--dry-run", type=int, default=0)
    ap.add_argument("--prompt-version", default=None, help="v1 | v2")
    ap.add_argument("--evaluators", default=None,
                    help="comma-separated evaluator names; default is all code evaluators")
    ap.add_argument("--repetitions", type=int, default=1,
                    help="run each example N times; needed before trusting a "
                         "signal delta, since the task is non-deterministic")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    # An *absent* variable means local development, where localhost is the right
    # default. A variable that is present but empty means something set it and
    # had nothing to set it to — an unconfigured CI secret. `setdefault` does not
    # replace an empty string, so the old default let that through, Phoenix fell
    # back to its own localhost:4317, and the run died 199 examples later in an
    # httpx stack trace instead of on the line that was actually wrong.
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if endpoint is None:
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
    elif not endpoint.strip():
        raise SystemExit(
            "PHOENIX_COLLECTOR_ENDPOINT is set but empty.\n"
            "In CI that means the PHOENIX_ENDPOINT secret is not configured.\n"
            "This experiment writes its results to Phoenix and compares them "
            "against the previous run, so it cannot run without one."
        )
    # Must be set before `agent.prompt` is imported — it resolves at import time.
    if args.prompt_version:
        os.environ["PROMPT_VERSION"] = args.prompt_version

    # `evals` always comes from this tree; `agent` may come from elsewhere.
    sys.path.insert(0, str(HERE))
    if args.agent_root:
        sys.path.insert(0, str(Path(args.agent_root).resolve()))

    from phoenix.otel import register
    register(project_name=f"travel-evals-{args.name}", auto_instrument=True, batch=True)

    import agent.loop as agent_loop
    from agent.prompt import PROMPT_VERSION
    print(f"agent under test: {agent_loop.__file__}")
    print(f"prompt version:   {PROMPT_VERSION}")

    from evals.agents.travel import evaluators, judges, truth  # noqa: F401  (registers)
    from evals.core import runner
    print(f"reference fixtures: {truth.DATA_DIR}")

    exp = runner.run(
        agent="travel",
        dataset_name=args.dataset,
        agent_fn=agent_loop.run_agent,
        experiment_name=args.name,
        names=[n.strip() for n in args.evaluators.split(",")] if args.evaluators else None,
        dry_run=args.dry_run or False,
        repetitions=args.repetitions,
    )
    print(f"\nexperiment_id={exp.get('experiment_id')}  project={exp.get('project_name')}")


if __name__ == "__main__":
    main()
