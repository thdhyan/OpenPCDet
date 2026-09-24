#!/usr/bin/env python3
"""Publish the G1 URDF on /robot_description so RViz's RobotModel display has
something to parse.

Isaac Sim's own ROS2 bridge publishes /tf directly from the USD - it never
publishes this topic, and nothing else in this pipeline does either, so
RobotModel silently renders empty (confirmed live 2026-08-11: ``ros2 topic
info /robot_description --verbose`` showed **zero publishers**, the topic
only existed because RViz's own subscription created it). Deliberately not
``robot_state_publisher`` - that node also republishes /tf from
joint_states, which would duplicate every frame Isaac Sim already publishes
directly from the USD.

    source /opt/ros/jazzy/setup.bash
    python3 scripts/publish_robot_description.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

REPO = Path(__file__).resolve().parent.parent
DEFAULT_URDF = (
    REPO / "assets/robot/g1_29/g1_29dof_with_hand_rev_1_0.urdf"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Latch the G1 URDF onto /robot_description.")
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF))
    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        raise SystemExit(f"URDF not found: {urdf_path}")

    # The URDF's mesh tags use bare relative paths ("meshes/pelvis.STL"),
    # which resolve fine for a package-aware URDF loader but not for RViz's
    # resource_retriever reading raw text off a topic - confirmed live
    # 2026-08-11 (RobotModel logged "Could not open file meshes/*.STL" for
    # every link). Rewriting to file:// URIs anchored at the URDF's own
    # directory makes them resolvable with no ROS package needed.
    urdf_text = urdf_path.read_text()
    mesh_dir = urdf_path.parent
    urdf_text = urdf_text.replace('filename="meshes/', f'filename="file://{mesh_dir}/meshes/')

    rclpy.init()
    node = Node("g1_robot_description_publisher")
    # transient_local so RViz picks it up even if RViz started before this
    # script did - matches how ROS's own robot_state_publisher latches it.
    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )
    pub = node.create_publisher(String, "/robot_description", qos)
    pub.publish(String(data=urdf_text))
    node.get_logger().info(f"published {urdf_path} on /robot_description (latched)")
    rclpy.spin(node)


if __name__ == "__main__":
    main()
