#!/usr/bin/env python3
"""Live check of the full G1 sensor suite while the WBC walks the robot.

Run against a sim started with --wbc-mode internal (ROS 2 env sourced):
    python3 scripts/check_sensor_suite.py

1. every sensor topic publishes at a sane rate (sim time: the sim runs
   slower than real time, so rates come from message stamps)
2. IMUs read gravity at rest (|a| ~ 9.81) with unit quaternions
3. /g1/cmd_vel forward for a few seconds moves the pelvis, robot stays up
Then re-run scripts/verify_sensor_tf.py to check the clouds after walking.
"""
from __future__ import annotations

import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, PointCloud2
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

TOPICS = {
    "/livox/mid360/points/a": (PointCloud2, 5.0),
    "/g1/camera/rgb": (Image, 5.0),
    "/g1/camera/depth": (Image, 5.0),
    "/g1/camera/depth/points": (PointCloud2, 5.0),
    "/g1/camera/depth/color/points": (PointCloud2, 3.0),
    "/g1/camera/camera_info": (CameraInfo, 5.0),
    "/g1/joint_states": (JointState, 20.0),
    "/g1/imu/pelvis": (Imu, 20.0),
    "/g1/imu/torso": (Imu, 20.0),
    "/livox/mid360/imu": (Imu, 20.0),
    "/g1/camera/imu": (Imu, 20.0),
    "/tf": (TFMessage, 20.0),
}
IMUS = [t for t, (ty, _) in TOPICS.items() if ty is Imu]


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("sensor_suite_check")
    buf = Buffer()
    _listener = TransformListener(buf, node)
    stamps: dict[str, list[float]] = {t: [] for t in TOPICS}
    last: dict[str, object] = {}

    def cb(topic):
        def f(msg):
            h = getattr(msg, "header", None) or msg.transforms[0].header
            stamps[topic].append(h.stamp.sec + h.stamp.nanosec * 1e-9)
            last[topic] = msg
        return f

    for topic, (ty, _) in TOPICS.items():
        node.create_subscription(ty, topic, cb(topic), qos_profile_sensor_data)
    cmd = node.create_publisher(Twist, "/g1/cmd_vel", 10)

    def spin(sec: float) -> None:
        end = time.monotonic() + sec
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)

    def pelvis():
        tf = buf.lookup_transform("World", "pelvis", rclpy.time.Time()).transform
        return np.array([tf.translation.x, tf.translation.y, tf.translation.z])

    spin(15.0)
    ok = True
    print("== rates (sim time) ==")
    for topic, (_, min_hz) in TOPICS.items():
        s = stamps[topic]
        s = sorted(set(s))
        hz = (len(s) - 1) / (s[-1] - s[0]) if len(s) > 2 and s[-1] > s[0] else 0.0
        good = hz >= min_hz * 0.5
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {topic:34s} {hz:6.1f} Hz (want >= {min_hz * 0.5:.1f})")

    print("== IMUs at rest ==")
    for topic in IMUS:
        m = last.get(topic)
        if m is None:
            ok = False
            print(f"  [FAIL] {topic}: no data")
            continue
        a = np.linalg.norm([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        q = np.linalg.norm([m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w])
        good = 8.8 < a < 10.8 and abs(q - 1) < 1e-3
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {topic:22s} |a|={a:5.2f} m/s^2  |q|={q:.4f}  frame={m.header.frame_id}")

    js = last.get("/g1/joint_states")
    n_joints = len(js.name) if js else 0
    ok &= n_joints == 29
    print(f"  [{'PASS' if n_joints == 29 else 'FAIL'}] joint_states: {n_joints} joints")

    def sim_now() -> float:
        return max(stamps["/g1/joint_states"])

    print("== WBC walk: cmd_vel vx=0.4 for 4 sim-seconds ==")
    p0 = pelvis()
    twist = Twist()
    twist.linear.x = 0.4
    t_end = sim_now() + 4.0
    while sim_now() < t_end:
        cmd.publish(twist)
        spin(0.05)
    cmd.publish(Twist())
    spin(2.0)
    p1 = pelvis()
    moved = float(np.linalg.norm((p1 - p0)[:2]))
    upright = p1[2] > 0.6
    ok &= moved > 0.8 and upright
    print(f"  [{'PASS' if moved > 0.8 else 'FAIL'}] pelvis moved {moved:.2f} m in xy (want > 0.8)")
    print(f"  [{'PASS' if upright else 'FAIL'}] pelvis z {p1[2]:.3f} m (upright)")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
