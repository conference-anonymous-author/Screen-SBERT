#!/bin/bash
set -e

require_image() {
    local image_tag="$1"

    if docker image inspect "${image_tag}" >/dev/null 2>&1; then
        echo "Found local image: ${image_tag}"
        return 0
    fi

    echo "Local image not found: ${image_tag}"
    echo "Build images first:"
    echo "  bash build.sh"
    exit 1
}

require_image "gui-parser_server:latest"
require_image "gui-parser_proxy:latest"

bash -c docker_scripts/run_server.sh
bash -c docker_scripts/run_proxy.sh
