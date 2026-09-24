# Custom Isaac Sim + Isaac ROS image (CuVSLAM + NVBlox)
# Extends nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1
#
# Isaac ROS 4.5 is used for the current dl driver line (CUDA 13.0 / 580+).
# Override ISAAC_ROS_RELEASE only after checking the host driver requirements.
#
# Build:
#   docker compose build

FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1

ARG DEBIAN_FRONTEND=noninteractive
ARG ISAAC_ROS_RELEASE=4.5
ENV DEBIAN_FRONTEND=noninteractive

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
# Isaac ROS 4.5 targets CUDA 13.0. Pin the apt solver so a fresh CUDA repo
# cannot silently pull 13.1+ packages that require a newer host driver.
COPY docker/isaac_ros_cuda_13_0.pref /etc/apt/preferences.d/g1-isaac-ros-cuda-13-0.pref
COPY docker/isaac_ros_tensorrt_13_0.pref /etc/apt/preferences.d/g1-isaac-ros-tensorrt-13-0.pref
COPY docker/isaac_ros_nccl_13_0.pref /etc/apt/preferences.d/g1-isaac-ros-nccl-13-0.pref
COPY docker/isaac_ros_opencv_4_6.pref /etc/apt/preferences.d/g1-isaac-ros-opencv-4-6.pref
# Isaac ROS NVBLOX needs the real CUDA 13 NPP libraries.
# Repo path is arch-specific: sbsa = ARM64 (Sparks), x86_64 = amd64.
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then CUDA_ARCH="sbsa"; else CUDA_ARCH="x86_64"; fi && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${CUDA_ARCH}/cuda-keyring_1.1-1_all.deb \
      -o /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends libnpp-13-0 && \
    rm -f /tmp/cuda-keyring.deb && \
    rm -rf /var/lib/apt/lists/*

# ── 1b. VPI4 apt repo (amd64 only) ───────────────────────────────────────────
# Isaac ROS needs libnvvpi4/vpi4-dev. On arm64 those come from the Isaac ROS
# repo itself (proven on spark2); on x86_64 NVIDIA publishes host VPI via the
# Jetson x86_64 repo - docs.nvidia.com/vpi/installation.html (Ubuntu 24.04).
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
      curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
        -o /usr/share/keyrings/nvidia-jetson-ota.asc && \
      echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson-ota.asc arch=amd64] https://repo.download.nvidia.com/jetson/x86_64/noble r39.2 main" \
        > /etc/apt/sources.list.d/nvidia-vpi.list && \
      apt-get update; \
    fi

# ── 2. Add Isaac ROS apt repo ────────────────────────────────────────────────
RUN mkdir -p /usr/share/keyrings /etc/apt/keyrings && \
    curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key -o /tmp/isaac-ros.key && \
    gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-isaac-ros.gpg /tmp/isaac-ros.key && \
    rm /tmp/isaac-ros.key && \
    ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then CODENAME="noble-fastos"; else CODENAME="noble"; fi && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-${ISAAC_ROS_RELEASE} $CODENAME main" \
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
