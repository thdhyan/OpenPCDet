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
# Ensure g1_sim package is importable (may be in G1_sim/g1_sim/ or parent dir)
if [ -d /workspace/thesis-sim/G1_sim/g1_sim ]; then
    export PYTHONPATH=/workspace/thesis-sim/G1_sim:$PYTHONPATH
fi

# CycloneDDS config (set here, not in docker-compose env, to avoid rclpy crash)
export CYCLONEDDS_URI=file:///root/.config/cyclonedds/cyclonedds.xml

# Isaac ROS environment (use Isaac Sim's bundled ROS 2)
# NOTE: Do NOT source /opt/ros/jazzy/setup.bash — it conflicts with Isaac Sim's ROS.
export ROS_PACKAGE_PATH=/opt/ros/isaac_ros_ws/src:$ROS_PACKAGE_PATH 2>/dev/null || true

echo "=== [entrypoint] GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi not available"

echo "=== [entrypoint] Installing onnxruntime-gpu ==="
/isaac-sim/python.sh -m pip install -q onnxruntime-gpu 2>&1 | tail -3

cd /workspace/thesis-sim/G1_sim

# Find g1_warehouse_sim.py — may be in G1_sim/ or G1_sim/scripts/
if [ -f scripts/g1_warehouse_sim.py ]; then
    SIM_SCRIPT=scripts/g1_warehouse_sim.py
    ASSET_DIR=scripts
elif [ -f g1_warehouse_sim.py ]; then
    SIM_SCRIPT=g1_warehouse_sim.py
    ASSET_DIR=.
else
    echo "ERROR: g1_warehouse_sim.py not found"
    exit 1
fi
echo "=== [entrypoint] Using script: $SIM_SCRIPT ==="

echo "=== [entrypoint] Starting Isaac Sim ==="
exec /isaac-sim/python.sh $SIM_SCRIPT \
    --config-dir assets/lidar_configs_rotary \
    --wbc-mode internal
