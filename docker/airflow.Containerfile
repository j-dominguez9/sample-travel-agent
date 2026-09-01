# Airflow with this project's eval dependencies baked in.
#
# The repo itself is bind-mounted rather than copied, so edits to a DAG or to
# evals/ show up on the next scheduler parse without a rebuild. That is right
# for learning and wrong for production: a real deployment installs a built
# wheel of this package into the image, which is what
# `[tool.hatch.build.targets.wheel] packages` in pyproject.toml is for.

FROM docker.io/apache/airflow:3.0.2-python3.12

# Airflow pins its own dependency tree, so let pip resolve against the
# constraints the image already satisfies rather than dragging in a conflicting
# pandas or httpx.
RUN pip install --no-cache-dir \
      "arize-phoenix-client>=3.3.0" \
      "arize-phoenix-evals>=3.5.1" \
      "arize-phoenix-otel>=0.17.1" \
      "openinference-instrumentation-anthropic>=2.1.1" \
      "anthropic>=1.2.0" \
      "python-dotenv" \
      "pyyaml"

# `evals` and `agent` are imported from the bind mount.
ENV PYTHONPATH=/opt/project
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/project/dags
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
# Each run reaches a real Phoenix and a real model. Without this, unpausing an
# hourly DAG fires a backlog of runs for every hour since start_date.
ENV AIRFLOW__SCHEDULER__CATCHUP_BY_DEFAULT=False
