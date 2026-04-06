#!/bin/bash
set -e

echo "========================================"
echo "Building GUI-Parser Docker Images"
echo "========================================"

bash docker_scripts/build_server.sh
bash docker_scripts/build_proxy.sh

echo ""
echo "Build complete:"
echo "  - gui-parser_server:latest"
echo "  - gui-parser_proxy:latest"
echo ""
