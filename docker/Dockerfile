# Custom Isaac Sim + Isaac ROS image (CUVSLAM + NVBLOX)
# Extends nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1
#
# Isaac Sim ships CUDA 12.8, but Isaac ROS release-4.6 needs CUDA 13 NPP.
# We install the real libnpp-13-0 from NVIDIA's CUDA repo and let the
# cu13 Python package provide libnvJitLink.so.13.
#
# Build:
#   docker compose build

FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1

ARG DEBIAN_FRONTEND=noninteractive

USER root

RUN mkdir -p /var/lib/apt/lists/partial

# ── 0. Add ROS 2 Jazzy apt repo (for PCL and other standard ROS packages) ────
# Do NOT install ros-jazzy-ros-base — it conflicts with Isaac Sim's bundled ROS.
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
      signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    rm -rf /var/lib/apt/lists/*

# ── 1. Add NVIDIA CUDA repo + install CUDA 13 NPP runtime ────────────────────
# Isaac ROS NVBLOX needs libnppidei.so.13, libnppim.so.13, libnppitc.so.13,
# libnppc.so.13. These are only in the real CUDA 13 NPP package.
# We also install cuda-cudart-13-0 (pulled as dep) which provides libcudart.so.13.
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
      -o /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends libnpp-13-0 && \
    rm -f /tmp/cuda-keyring.deb && \
    rm -rf /var/lib/apt/lists/*

# ── 2. Add Isaac ROS apt repo ────────────────────────────────────────────────
RUN mkdir -p /usr/share/keyrings /etc/apt/keyrings && \
    curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key -o /tmp/isaac-ros.key && \
    gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-isaac-ros.gpg /tmp/isaac-ros.key && \
    rm /tmp/isaac-ros.key && \
    ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then CODENAME="noble-fastos"; else CODENAME="noble"; fi && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-4.6 $CODENAME main" \
      > /etc/apt/sources.list.d/nvidia-isaac-ros.list && \
    apt-get update

# ── 3. Install Isaac ROS packages (no ros-jazzy-ros-base) ────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-jazzy-isaac-ros-visual-slam \
      ros-jazzy-isaac-ros-nvblox \
      ros-jazzy-nav2-costmap-2d \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-tf2-ros \
      ros-jazzy-tf2-geometry-msgs \
      ros-jazzy-pcl-conversions \
      ros-jazzy-pcl-ros \
    && rm -rf /var/lib/apt/lists/*

# ── 4. ldconfig for CUDA 13 libs in non-standard paths ───────────────────────
# cu13 Python package has libnvJitLink.so.13 (needed by CUVSLAM)
RUN echo "/isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/cu13/lib" \
      > /etc/ld.so.conf.d/cuda13-cu13.conf && \
    ldconfig

# ── 5. CycloneDDS config ─────────────────────────────────────────────────────
# CycloneDDS config is handled by entrypoint.sh (copies from volume mount at runtime)

# ── 6. Environment setup ─────────────────────────────────────────────────────
RUN echo "" >> /root/.bashrc && \
    echo "# Isaac Sim ROS2 core" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:\$LD_LIBRARY_PATH" >> /root/.bashrc && \
    echo "export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/rclpy:\$PYTHONPATH" >> /root/.bashrc
