# Custom Isaac Sim + Isaac ROS image (CUVSLAM + NVBLOX)
# Extends nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1
#
# Uses dummy metapackages for CUDA deps and installs Isaac ROS from apt
# WITHOUT ros-jazzy-ros-base (which conflicts with Isaac Sim's bundled ROS).
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

# ── 1. Dummy CUDA/TensorRT packages ─────────────────────────────────────────
RUN apt-get update && apt-get install -y equivs gnupg && \
    mkdir -p /tmp/dummy && cd /tmp/dummy && \
    for pkg in \
      "cuda-toolkit-13-0 13.0.0-1 all" \
      "cuda-toolkit-13-0-1 13.0.0-1 all" \
      "cuda-cudart-dev-13-0 13.0.0-1 all" \
      "libcudnn9-cuda-13 9.0.0-1 all" \
      "tensorrt 10.0.0-1 all" \
      "triton-server 2.0.0-1 all" \
    ; do \
      name=$(echo "$pkg" | cut -d' ' -f1); \
      ver=$(echo "$pkg" | cut -d' ' -f2); \
      arch=$(echo "$pkg" | cut -d' ' -f3); \
      printf "Package: %s\nVersion: %s\nArchitecture: %s\nDescription: Dummy for Isaac ROS compat\n" "$name" "$ver" "$arch" > "${name}.cfg" && \
      equivs-build "${name}.cfg"; \
    done && \
    dpkg -i *.deb && \
    cd / && rm -rf /tmp/dummy && \
    apt-get purge -y equivs && \
    apt-get autoremove -y 2>/dev/null || true && \
    rm -rf /var/lib/apt/lists/*

# ── 2. Add Isaac ROS apt repo ────────────────────────────────────────────────
# Download key, dearmor into keyring, and add repo
RUN mkdir -p /usr/share/keyrings /etc/apt/keyrings && \
    curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key -o /tmp/isaac-ros.key && \
    gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-isaac-ros.gpg /tmp/isaac-ros.key && \
    rm /tmp/isaac-ros.key && \
    ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then CODENAME="noble-fastos"; else CODENAME="noble"; fi && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-4.6 $CODENAME main" \
      > /etc/apt/sources.list.d/nvidia-isaac-ros.list && \
    apt-get update

# ── 3. Install ONLY the Isaac ROS packages we need (no ros-jazzy-ros-base) ───
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

# ── 4. CycloneDDS config ─────────────────────────────────────────────────────
COPY docker/cyclonedds_isaac_ros.xml /root/.config/cyclonedds/cyclonedds.xml

# ── 5. Environment setup ─────────────────────────────────────────────────────
RUN echo "" >> /root/.bashrc && \
    echo "# Isaac Sim ROS2 core" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:\$LD_LIBRARY_PATH" >> /root/.bashrc && \
    echo "export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/rclpy:\$PYTHONPATH" >> /root/.bashrc
