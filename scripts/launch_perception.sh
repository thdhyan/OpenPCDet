#!/bin/bash
# Launch the CUVSLAM + NVBLOX perception stack inside the Isaac Sim container.
#
# Usage (from the laptop or any machine with SSH access to spark2):
#   ssh aim_spark02 'docker exec isaac-sim-ros bash /workspace/thesis-sim/G1_sim/scripts/launch_perception.sh'
#
# Or from inside the container:
#   bash /workspace/thesis-sim/G1_sim/scripts/launch_perception.sh
#
# Prerequisites:
#   - Isaac Sim container running with sim started
#   - zenoh bridge running (container-side and laptop-side)

set -euo pipefail

# Source Isaac Sim's bundled ROS 2 (NOT /opt/ros/jazzy/setup.bash)
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/rclpy:${PYTHONPATH:-}
export PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/bin:${PATH:-}
export CYCLONEDDS_URI=file:///root/.config/cyclonedds/cyclonedds.xml

# Add apt-installed ROS packages to path (if installed)
if [ -d /opt/ros/jazzy ]; then
    export PATH=/opt/ros/jazzy/bin:${PATH:-}
    export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:${PYTHONPATH:-}
fi

echo "=== CUVSLAM + NVBLOX Perception Stack ==="
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "RMW=$RMW_IMPLEMENTATION"

# Launch the perception stack
exec ros2 launch /workspace/thesis-sim/G1_sim/launch/perception_launch.py
