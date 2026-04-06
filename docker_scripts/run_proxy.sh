#!/bin/bash
set -e

IMAGE_TAG="gui-parser_proxy:latest"

PROXY_PORT="${PROXY_PORT:-4023}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-4001}"
TRITON_GRPC_URL="${TRITON_GRPC_URL:-gui_parser_server:${TRITON_GRPC_PORT}}"

echo "========================================"
echo "Running GUI-Parser Proxy Docker Container"
echo "========================================"
echo "Image:       ${IMAGE_TAG}"
echo "Host Port:   ${PROXY_PORT}"
echo "Container:   4023"
echo "Triton gRPC: ${TRITON_GRPC_URL}"
echo "========================================"

if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Local image not found: ${IMAGE_TAG}"
    echo "This script does not pull from remote registries."
    echo "Run ./run.sh or docker_scripts/build_proxy.sh first."
    exit 1
fi

# Create the screen_sbert network if it does not exist
docker network ls | grep -q screen_sbert || docker network create screen_sbert >/dev/null

# Stop and remove the existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q '^gui_parser_proxy$'; then
    echo "Stopping and removing existing container..."
    docker stop gui_parser_proxy >/dev/null 2>&1 || true
    docker rm gui_parser_proxy >/dev/null 2>&1 || true
fi

# Create a new container
docker container create --net=screen_sbert --restart unless-stopped \
    -e TRITON_GRPC_URL="${TRITON_GRPC_URL}" \
    --name gui_parser_proxy \
    -p${PROXY_PORT}:4023 \
    ${IMAGE_TAG} >/dev/null

docker container start gui_parser_proxy >/dev/null

echo ""
echo "Container started: gui_parser_proxy"
echo ""
