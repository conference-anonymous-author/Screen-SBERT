#!/bin/bash
set -e

if docker ps -a --format '{{.Names}}' | grep -q '^gui_parser_proxy$'; then
    echo "Stopping and removing existing gui_parser_proxy container..."
    docker stop gui_parser_proxy >/dev/null 2>&1 || true
    docker rm gui_parser_proxy >/dev/null 2>&1 || true
fi

if docker ps -a --format '{{.Names}}' | grep -q '^gui_parser_server$'; then
    echo "Stopping and removing existing gui_parser_server container..."
    docker stop gui_parser_server >/dev/null 2>&1 || true
    docker rm gui_parser_server >/dev/null 2>&1 || true
fi

echo "All services have been stopped and removed."
