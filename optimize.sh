#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

OPTIMIZE_IMAGE="${OPTIMIZE_IMAGE:-nvcr.io/nvidia/tritonserver:25.02-py3}"
OPTIMIZE_SHM_SIZE="${OPTIMIZE_SHM_SIZE:-1g}"
OPTIMIZE_MODELS_DIR="${OPTIMIZE_MODELS_DIR:-server/models}"
OPTIMIZE_SCRIPT_DIR="${OPTIMIZE_SCRIPT_DIR:-server/optimize}"
OPTIMIZE_USE_GPU="${OPTIMIZE_USE_GPU:-true}"
OPTIMIZE_GPU_DEVICE="${OPTIMIZE_GPU_DEVICE:-all}"

to_abs_path() {
    local p="$1"
    if [[ "${p}" = /* ]]; then
        echo "${p}"
    else
        echo "${PROJECT_ROOT}/${p#./}"
    fi
}

OPTIMIZE_MODELS_DIR="$(to_abs_path "${OPTIMIZE_MODELS_DIR}")"
OPTIMIZE_SCRIPT_DIR="$(to_abs_path "${OPTIMIZE_SCRIPT_DIR}")"

if [ ! -d "${OPTIMIZE_MODELS_DIR}" ]; then
    echo "Models directory not found: ${OPTIMIZE_MODELS_DIR}"
    exit 1
fi

if [ ! -f "${OPTIMIZE_SCRIPT_DIR}/optimize.sh" ]; then
    echo "optimize.sh not found in: ${OPTIMIZE_SCRIPT_DIR}"
    exit 1
fi

GPU_ARGS=()
if [ "${OPTIMIZE_USE_GPU}" = "true" ]; then
    GPU_ARGS=(--gpus="${OPTIMIZE_GPU_DEVICE}")
fi

docker run --shm-size="${OPTIMIZE_SHM_SIZE}" "${GPU_ARGS[@]}" --rm -it \
    -v "${OPTIMIZE_MODELS_DIR}:/models" \
    -v "${OPTIMIZE_SCRIPT_DIR}:/optimize" \
    "${OPTIMIZE_IMAGE}" \
    /optimize/optimize.sh
