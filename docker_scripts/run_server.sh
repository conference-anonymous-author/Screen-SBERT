#!/bin/bash
set -e

IMAGE_TAG="gui-parser_server:latest"

TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-4000}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-4001}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-4002}"

detect_gpu_count() {
    local count
    count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -z "${count}" ]] || ! [[ "${count}" =~ ^[0-9]+$ ]]; then
        echo "0"
        return
    fi
    echo "${count}"
}

patch_model_gpu_instance_group() {
    local config_path="$1"
    local gpu_id="$2"

    python3 - "$config_path" "$gpu_id" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
gpu_id = sys.argv[2]
text = path.read_text(encoding="utf-8")

block = (
    "instance_group [\n"
    "    {\n"
    "        kind: KIND_GPU\n"
    "        count: 1\n"
    f"        gpus: [{gpu_id}]\n"
    "    }\n"
    "]\n"
)

lines = text.splitlines()
start = -1
for i, line in enumerate(lines):
    if line.strip().startswith("instance_group"):
        start = i
        break

if start >= 0:
    depth = 0
    end = -1
    for i in range(start, len(lines)):
        depth += lines[i].count("[")
        depth -= lines[i].count("]")
        if depth <= 0 and i > start:
            end = i
            break
    if end < 0:
        raise SystemExit(f"Failed to locate end of instance_group block in {path}")
    new_lines = lines[:start] + block.rstrip("\n").splitlines() + lines[end + 1 :]
    text = "\n".join(new_lines).rstrip() + "\n"
else:
    text = text.rstrip() + "\n\n" + block

path.write_text(text, encoding="utf-8")
PY
}

echo "========================================"
echo "Running GUI-Parser Server Docker Container"
echo "========================================"
echo "Image:       ${IMAGE_TAG}"
echo "HTTP Port:   ${TRITON_HTTP_PORT}"
echo "gRPC Port:   ${TRITON_GRPC_PORT}"
echo "Metrics:     ${TRITON_METRICS_PORT}"
echo "========================================"

if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Local image not found: ${IMAGE_TAG}"
    echo "This script does not pull from remote registries."
    echo "Run ./run.sh or docker_scripts/build_server.sh first."
    exit 1
fi

GPU_COUNT="$(detect_gpu_count)"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
    echo "No NVIDIA GPU detected via nvidia-smi. GUI-Parser server requires GPU."
    exit 1
fi

FLORENCE_GPU_ID=0
if [[ "${GPU_COUNT}" -ge 2 ]]; then
    SIGLIP_GPU_ID=1
else
    SIGLIP_GPU_ID=0
fi

echo "Detected GPUs: ${GPU_COUNT} (florence2->GPU ${FLORENCE_GPU_ID}, siglip_engine->GPU ${SIGLIP_GPU_ID})"

# Create the screen_sbert network if it does not exist
docker network ls | grep -q screen_sbert || docker network create screen_sbert >/dev/null

# Stop and remove the existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q '^gui_parser_server$'; then
    echo "Stopping and removing existing container..."
    docker stop gui_parser_server >/dev/null 2>&1 || true
    docker rm gui_parser_server >/dev/null 2>&1 || true
fi

# Create a new container
docker container create --gpus all --net=screen_sbert --restart unless-stopped \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-p${TRITON_HTTP_PORT}:${TRITON_HTTP_PORT} \
		-p${TRITON_GRPC_PORT}:${TRITON_GRPC_PORT} \
		-p${TRITON_METRICS_PORT}:${TRITON_METRICS_PORT} \
		--name gui_parser_server ${IMAGE_TAG} \
    /opt/tritonserver/bin/tritonserver \
    --model-repository=/models \
    --model-control-mode=explicit \
    --load-model="*" \
    --http-port=${TRITON_HTTP_PORT} \
    --grpc-port=${TRITON_GRPC_PORT} \
    --metrics-port=${TRITON_METRICS_PORT} \
    --cuda-memory-pool-byte-size=0:134217728 >/dev/null

if [ -d "./server" ]; then
    TMP_SERVER_DIR="$(mktemp -d)"
    trap 'rm -rf "${TMP_SERVER_DIR}"' EXIT

    cp -r ./server/models "${TMP_SERVER_DIR}/models"
    patch_model_gpu_instance_group "${TMP_SERVER_DIR}/models/florence2/config.pbtxt" "${FLORENCE_GPU_ID}"
    patch_model_gpu_instance_group "${TMP_SERVER_DIR}/models/siglip_engine/config.pbtxt" "${SIGLIP_GPU_ID}"

    docker cp "${TMP_SERVER_DIR}/models" gui_parser_server:/
    docker cp ./server/utils.py gui_parser_server:/utils.py
fi
docker container start gui_parser_server >/dev/null

echo ""
echo "Container started: gui_parser_server"
echo ""
