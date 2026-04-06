#!/bin/bash
set -e

docker build --pull=false -t gui-parser_proxy:latest proxy/
