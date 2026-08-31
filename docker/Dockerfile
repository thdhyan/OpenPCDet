# Custom Isaac Sim + Isaac ROS image (CUVSLAM + NVBLOX)
# Extends nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1
#
# Build:
#   docker compose build
#
# The base image already has:
#   - Isaac Sim + ROS 2 Jazzy (cyclone/rmw_cyclonedds_cpp)
#   - CUDA 13.0, Python 3.12
#   - Isaac Sim ROS2 core bindings

FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1

ARG DEBIAN_FRONTEND=noninteractive

# ── 1. Add ROS 2 Jazzy apt repo (needed for base ROS packages) ──────────────
RUN apt-get update && apt-get install -y curl gnupg lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
      signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-jazzy-ros-base \
      python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Add Isaac ROS apt repo ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y curl gnupg && \
    mkdir -p /usr/share/keyrings && \
    curl -fsSL https://isaac.download.nvidia.com/isaac-ros/release-4.6/keyring.gpg \
      -o /usr/share/keyrings/isaac-ros-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
      signed-by=/usr/share/keyrings/isaac-ros-keyring.gpg] \
      https://isaac.download.nvidia.com/isaac-ros/release-4.6 noble main" \
      > /etc/apt/sources.list.d/isaac-ros.list && \
    apt-get update

# ── 3. Install Isaac ROS CUVSLAM + NVBLOX + dependencies ────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-jazzy-isaac-ros-visual-slam \
      ros-jazzy-isaac-ros-nvblox \
      ros-jazzy-nav2-costmap-2d \
      ros-jazzy-nav2-bringup \
      ros-jazzy-rviz2 \
      ros-jazzy-tf2-tools \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-joint-state-publisher-gui \
      ros-jazzy-xacro \
      ros-jazzy-tf2-ros \
      ros-jazzy-tf2-geometry-msgs \
      ros-jazzy-pcl-conversions \
      ros-jazzy-pcl-ros \
    && rm -rf /var/lib/apt/lists/*

# ── 4. CycloneDDS config (inside container) ──────────────────────────────────
COPY docker/cyclonedds_isaac_ros.xml /root/.config/cyclonedds/cyclonedds.xml

# ── 5. Source Isaac ROS + Isaac Sim in bashrc ────────────────────────────────
RUN echo "" >> /root/.bashrc && \
    echo "# Isaac ROS" >> /root/.bashrc && \
    echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# Isaac Sim ROS2 core" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:\$LD_LIBRARY_PATH" >> /root/.bashrc && \
    echo "export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/rclpy:\$PYTHONPATH" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# Isaac ROS environment activation" >> /root/.bashrc && \
    echo 'if [ -f /opt/ros/jazzy/share/isaac_ros_visual_slam/setup.bash ]; then source /opt/ros/jazzy/share/isaac_ros_visual_slam/setup.bash; fi' >> /root/.bashrc

# ── 6. Verify installation ──────────────────────────────────────────────────
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    echo '=== Isaac ROS packages installed ===' && \
    ros2 pkg list 2>/dev/null | grep isaac_ros && \
    echo '=== CUVSLAM ===' && \
    ros2 pkg list 2>/dev/null | grep -i cuvslam && \
    echo '=== NVBLOX ===' && \
    ros2 pkg list 2>/dev/null | grep -i nvblox"
