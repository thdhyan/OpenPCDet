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

# CycloneDDS config — prefer the volume-mounted copy (allows updates without rebuild)
mkdir -p /root/.config/cyclonedds
if [ -f /workspace/thesis-sim/docker/cyclonedds_isaac_ros.xml ]; then
    cp /workspace/thesis-sim/docker/cyclonedds_isaac_ros.xml /root/.config/cyclonedds/cyclonedds.xml
fi
export CYCLONEDDS_URI=file:///root/.config/cyclonedds/cyclonedds.xml

# CUDA 13 libraries (NPP for NVBLOX, cudart for CUVSLAM)
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

# Isaac ROS environment (use Isaac Sim's bundled ROS 2 for rclcpp,
# apt-installed /opt/ros/jazzy for Isaac ROS composable nodes)
export AMENT_PREFIX_PATH=/opt/ros/jazzy
export ROS_PACKAGE_PATH=/opt/ros/jazzy/share:${ROS_PACKAGE_PATH:-}
export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:${PYTHONPATH:-}
export PATH=/opt/ros/jazzy/bin:${PATH:-}

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
