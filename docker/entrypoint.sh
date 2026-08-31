#!/bin/bash
# Entrypoint for the Isaac Sim + ROS2 + Isaac ROS container.
# Installs onnxruntime-gpu and starts the sim.
# Isaac ROS (CUVSLAM + NVBLOX) is baked into the Docker image.
#
# Portable across DL, Spark, and other NVIDIA GPU servers.

export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ISAAC_SIM_ROOT=/isaac-sim
export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/rclpy:$PYTHONPATH

# CycloneDDS config (set here, not in docker-compose env, to avoid rclpy crash)
export CYCLONEDDS_URI=file:///root/.config/cyclonedds/cyclonedds.xml

# Isaac ROS environment
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

echo "=== [entrypoint] GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi not available"

echo "=== [entrypoint] Installing onnxruntime-gpu ==="
/isaac-sim/python.sh -m pip install -q onnxruntime-gpu 2>&1 | tail -3

cd /workspace/thesis-sim/G1_sim

echo "=== [entrypoint] Starting Isaac Sim ==="
exec /isaac-sim/python.sh scripts/g1_warehouse_sim.py \
    --config-dir assets/lidar_configs_rotary \
    --wbc-mode internal
