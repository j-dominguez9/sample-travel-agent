#!/usr/bin/env bash
# Run Airflow locally with this repo's DAGs, to explore the feedback loop in a UI.
#
#   ./scripts/airflow_dev.sh           # build if needed, then start
#   ./scripts/airflow_dev.sh build     # rebuild the image
#   ./scripts/airflow_dev.sh stop      # stop and remove the container
#   ./scripts/airflow_dev.sh reset     # also delete the Airflow database
#   ./scripts/airflow_dev.sh logs      # follow logs
#   ./scripts/airflow_dev.sh shell     # a shell inside the container
#
# Airflow runs in a container rather than in .venv on purpose: its dependency
# tree is large and version-pinned and would fight this project's pandas. More
# to the point, Airflow is the customer's runtime, not our dependency — we ship
# DAG files plus the `evals` package, which is what the container consumes.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="travel-agent-airflow:3.0.2"
CONTAINER="travel-agent-airflow"
VOLUME="travel-agent-airflow-home"
PORT="${AIRFLOW_PORT:-8080}"

# This repo is often edited from inside a Fedora toolbx, where podman lives on
# the host. flatpak-spawn hops out; outside a toolbox it is a no-op prefix.
if [[ -f /run/.toolboxenv ]] && command -v flatpak-spawn >/dev/null; then
  PODMAN=(flatpak-spawn --host podman)
else
  PODMAN=(podman)
fi

build() {
  echo "==> building $IMAGE"
  "${PODMAN[@]}" build -t "$IMAGE" -f "$REPO/docker/airflow.Containerfile" "$REPO"
}

stop() {
  "${PODMAN[@]}" rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

case "${1:-start}" in
  build) build; exit 0 ;;
  stop)  stop; echo "stopped"; exit 0 ;;
  logs)  exec "${PODMAN[@]}" logs -f "$CONTAINER" ;;
  shell) exec "${PODMAN[@]}" exec -it "$CONTAINER" bash ;;
  reset)
    stop
    "${PODMAN[@]}" volume rm "$VOLUME" >/dev/null 2>&1 || true
    echo "database removed"
    ;;
esac

"${PODMAN[@]}" image exists "$IMAGE" || build

[[ -f "$REPO/.env" ]] || { echo "missing $REPO/.env (needs ANTHROPIC_API_KEY)"; exit 1; }
stop

# Phoenix runs on the host. From inside the container that is
# host.containers.internal, which podman wires to the host gateway.
echo "==> starting $CONTAINER on port $PORT"
"${PODMAN[@]}" run -d \
  --name "$CONTAINER" \
  -p "${PORT}:8080" \
  --add-host=host.containers.internal:host-gateway \
  -v "$REPO:/opt/project:ro,Z" \
  -v "$VOLUME:/opt/airflow" \
  --env-file "$REPO/.env" \
  -e PHOENIX_COLLECTOR_ENDPOINT="http://host.containers.internal:6006" \
  -e AIRFLOW__API__EXPOSE_CONFIG=True \
  "$IMAGE" \
  standalone >/dev/null

cat <<EOF

  UI          http://127.0.0.1:${PORT}
  user        admin
  password    ./scripts/airflow_dev.sh logs   (printed on first boot), or
              podman exec ${CONTAINER} cat /opt/airflow/simple_auth_manager_passwords.json.generated

  Both DAGs start paused, which is deliberate — each run calls a real model.
  Unpause one in the UI, or trigger a single run:

    ${PODMAN[*]} exec ${CONTAINER} airflow dags trigger agent_monitoring_travel

  DAGs are bind-mounted from ${REPO}/dags, so edits appear on the next parse.

EOF
