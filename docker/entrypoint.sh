#!/bin/bash
# Entrypoint for the Isaac Sim + ROS2 container.
# Installs dependencies, sets up Isaac ROS (CUVSLAM + NVBLOX), and starts the sim.
# Portable across DL, Spark, and other NVIDIA GPU servers.
#
# NOTE: No 'set -e' — we need the sim to launch even if Isaac ROS install fails.
# The sim works without CUVSLAM/NVBLOX; they are optional enhancements.

export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ISAAC_SIM_ROOT=/isaac-sim
export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/rclpy:$PYTHONPATH
export ISAAC_ROS_WS=/workspace/isaac_ros_ws

echo "=== [entrypoint] GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi not available"

echo "=== [entrypoint] Installing onnxruntime-gpu ==="
/isaac-sim/python.sh -m pip install -q onnxruntime-gpu 2>&1 | tail -3

# ── Isaac ROS (CUVSLAM + NVBLOX) — best-effort install ──────────────────
echo "=== [entrypoint] Setting up Isaac ROS ==="
(
    # Subshell so failures don't kill the main entrypoint

    # Install base tools (use sudo since container runs as non-root)
    sudo apt-get update -qq 2>/dev/null || true
    sudo apt-get install -y -qq curl gnupg lsb-release git git-lfs 2>/dev/null || true
    git lfs install 2>/dev/null || true

    # Add NVIDIA Isaac ROS key + repo
    sudo mkdir -p /usr/share/keyrings
    curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-isaac-ros.gpg 2>/dev/null || true

    echo "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] \
https://isaac.download.nvidia.com/isaac-ros/release-4.6 noble main" \
        | sudo tee /etc/apt/sources.list.d/nvidia-isaac-ros.list > /dev/null

    # Add ROS2 Jazzy repo
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg 2>/dev/null || true

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu noble main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    sudo apt-get update -qq 2>/dev/null || true

    # Install CUVSLAM (Visual SLAM)
    echo "[entrypoint] Installing CUVSLAM..."
    if apt-cache show ros-jazzy-isaac-ros-visual-slam >/dev/null 2>&1; then
        sudo apt-get install -y -qq ros-jazzy-isaac-ros-visual-slam 2>&1 | tail -3
        echo "[entrypoint] CUVSLAM installed from Debian"
    else
        echo "[entrypoint] CUVSLAM Debian unavailable — building from source"
        sudo mkdir -p $ISAAC_ROS_WS/src
        cd $ISAAC_ROS_WS/src
        git clone -b release-4.6 --depth 1 \
            https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git 2>&1 | tail -3
        cd $ISAAC_ROS_WS
        sudo apt-get install -y -qq python3-colcon-common-extensions 2>/dev/null || true
        colcon build --symlink-install --packages-up-to isaac_ros_visual_slam 2>&1 | tail -5
    fi

    # Install NVBLOX (3D mapping)
    echo "[entrypoint] Installing NVBLOX..."
    if apt-cache show ros-jazzy-isaac-ros-nvblox >/dev/null 2>&1; then
        sudo apt-get install -y -qq ros-jazzy-isaac-ros-nvblox 2>&1 | tail -3
        echo "[entrypoint] NVBLOX installed from Debian"
    else
        echo "[entrypoint] NVBLOX Debian unavailable — building from source"
        sudo mkdir -p $ISAAC_ROS_WS/src
        cd $ISAAC_ROS_WS/src
        git clone -b release-4.6 --depth 1 \
            https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox.git 2>&1 | tail -3
        cd $ISAAC_ROS_WS
        colcon build --symlink-install --packages-up-to isaac_ros_nvblox 2>&1 | tail -5
    fi

    echo "=== [entrypoint] Isaac ROS setup complete ==="
    sudo rm -rf /var/lib/apt/lists/* 2>/dev/null || true
) 2>&1 | tail -20  # Show last 20 lines of Isaac ROS install output
# ── end Isaac ROS install ────────────────────────────────────────────────

cd /workspace/thesis-sim/G1_sim

echo "=== [entrypoint] Starting Isaac Sim ==="
exec /isaac-sim/python.sh scripts/g1_warehouse_sim.py \
    --config-dir assets/lidar_configs_rotary \
    --wbc-mode internal
