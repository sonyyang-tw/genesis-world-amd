#!/bin/bash
# Launches an interactive shell in the genesis-world-amd container.
# The repo root (parent of this script's directory) is mounted into the container.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CONTAINER_NAME="genesis-world-amd-shell"
IMAGE_NAME="genesis-world-amd"
PORT=8802

if docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo "Removing existing ${CONTAINER_NAME}..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1
    docker rm   "$CONTAINER_NAME" >/dev/null 2>&1
fi

echo "Starting interactive shell in ${CONTAINER_NAME}..."
echo "Mounting repo root: ${REPO_ROOT}"
echo "(Port ${PORT} is also forwarded — you can run \`jupyter lab --ip=0.0.0.0 --port=${PORT}\` from inside if needed.)"
echo

docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --privileged \
    --group-add dialout \
    --group-add video \
    --ipc=host \
    --shm-size=8g \
    --device=/dev/kfd \
    --device=/dev/dri \
    -p ${PORT}:${PORT} \
    -v "$REPO_ROOT":/opt/workspace/genesis-world \
    -w /opt/workspace/genesis-world \
    --entrypoint bash \
    "$IMAGE_NAME"
