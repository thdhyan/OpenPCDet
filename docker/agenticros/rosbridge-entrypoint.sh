#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/jazzy/setup.bash

readonly PARAMS=/etc/rosbridge/read_only_params.yaml
ros2 run rosapi rosapi_node --ros-args --params-file "$PARAMS" &
api_pid=$!
bridge_pid=""

cleanup() {
  [[ -n "$bridge_pid" ]] && kill "$bridge_pid" "$api_pid" 2>/dev/null || true
  wait "$api_pid" 2>/dev/null || true
  [[ -n "$bridge_pid" ]] && wait "$bridge_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ros2 run rosbridge_server rosbridge_websocket --ros-args --params-file "$PARAMS" &
bridge_pid=$!

wait -n "$api_pid" "$bridge_pid"
