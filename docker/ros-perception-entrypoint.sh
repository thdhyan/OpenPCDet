#!/usr/bin/env bash
set -euo pipefail

# The Isaac Sim container owns the DDS graph. This sidecar only runs Isaac
# ROS nodes and joins the same host-network CycloneDDS domain.
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ERROR: /opt/ros/jazzy/setup.bash is missing from the Isaac ROS image" >&2
  exit 1
fi
source /opt/ros/jazzy/setup.bash

export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export AMENT_PREFIX_PATH="/opt/ros/jazzy${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export ROS_PACKAGE_PATH="/opt/ros/jazzy/share${ROS_PACKAGE_PATH:+:$ROS_PACKAGE_PATH}"
export LD_LIBRARY_PATH="/isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-13.0/lib64:/opt/ros/jazzy/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/workspace/thesis-sim${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p /root/.config/cyclonedds
cp /workspace/thesis-sim/docker/cyclonedds_isaac_ros.xml /root/.config/cyclonedds/cyclonedds.xml
export CYCLONEDDS_URI=file:///root/.config/cyclonedds/cyclonedds.xml

exec "$@"
