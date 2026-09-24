#!/usr/bin/env python3
"""Live check of the full G1 sensor suite while the WBC walks the robot.

Run against a sim started with --wbc-mode internal (ROS 2 env sourced):
    python3 scripts/check_sensor_suite.py

1. every sensor topic publishes at a sane rate (sim time: the sim runs
   slower than real time, so rates come from message stamps)
2. IMUs read gravity at rest (|a| ~ 9.81) with unit quaternions
3. 29 body + 14 Dex3 joints; /g1/arm_cmd and /g1/hand_cmd targets are tracked
4. /g1/cmd_vel backward (the robot faces the packing table) moves the pelvis,
   robot stays up
Then re-run scripts/verify_sensor_tf.py to check the clouds after walking.
"""
from __future__ import annotations

import time

import numpy as np
import rclpy
import re
from pathlib import Path

from geometry_msgs.msg import Twist, WrenchStamped
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
TOPICS.update({
    f"/g1/dex3/{side}/{finger}/contact": (WrenchStamped, 20.0)
    for side in ("left", "right") for finger in ("thumb", "index", "middle")
})
IMUS = [t for t, (ty, _) in TOPICS.items() if ty is Imu]
URDF = Path(__file__).resolve().parents[1] / "assets/robot/g1_29/g1_29dof_with_hand_rev_1_0.urdf"
DEX3 = [f"{s}_hand_{f}_joint" for s in ("left", "right")
        for f in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]


def joint_limits() -> dict[str, tuple[float, float]]:
    text = URDF.read_text()
    out = {}
    for name, body in re.findall(r'<joint name="([^"]+)" type="revolute">(.*?)</joint>', text, re.S):
        m = re.search(r'<limit[^>]*lower="([-\d.e]+)"[^>]*upper="([-\d.e]+)"', body)
        if m:
            out[name] = (float(m.group(1)), float(m.group(2)))
    return out


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
    arm_pub = node.create_publisher(JointState, "/g1/arm_cmd", 10)
    hand_pub = node.create_publisher(JointState, "/g1/hand_cmd", 10)

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
    names = list(js.name) if js else []
    n_hand = sum(n in names for n in DEX3)
    good = len(names) == 43 and n_hand == 14
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] joint_states: {len(names)} joints ({n_hand} Dex3)")

    tf_msg = last.get("/tf")
    tf_children = {transform.child_frame_id for transform in tf_msg.transforms} if tf_msg else set()
    dex3_frames = {
        "left_hand_palm_link", "right_hand_palm_link",
        "left_hand_thumb_2_link", "right_hand_thumb_2_link",
        "left_hand_index_1_link", "right_hand_index_1_link",
        "left_hand_middle_1_link", "right_hand_middle_1_link",
    }
    good = dex3_frames <= tf_children
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] Dex3 TF frames: {len(dex3_frames & tf_children)}/{len(dex3_frames)}")

    def sim_now() -> float:
        return max(stamps["/g1/joint_states"])

    def joints_now(names_wanted):
        m = last["/g1/joint_states"]
        idx = {n: i for i, n in enumerate(m.name)}
        return np.array([m.position[idx[n]] for n in names_wanted])

    def track(pub, names_wanted, target, label, tol, secs=3.0):
        nonlocal ok
        msg = JointState(name=list(names_wanted), position=[float(v) for v in target])
        t_end = sim_now() + secs
        while sim_now() < t_end:
            pub.publish(msg)
            spin(0.05)
        err = np.abs(joints_now(names_wanted) - target)
        good = bool(err.max() < tol)
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: max |q - target| = {err.max():.3f} rad (tol {tol})")

    print("== Dex3 control: /g1/hand_cmd ==")
    limits = joint_limits()
    closed = np.array([0.5 * (lo if abs(lo) > abs(hi) else hi) for lo, hi in (limits[n] for n in DEX3)])
    track(hand_pub, DEX3, closed, "fingers curl to 50% of range", 0.15)
    track(hand_pub, DEX3, np.zeros(len(DEX3)), "fingers reopen to 0", 0.15)

    print("== Arm control: /g1/arm_cmd ==")
    arms = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_elbow_joint", "right_elbow_joint"]
    q0 = joints_now(arms)
    track(arm_pub, arms, q0 + np.array([-0.3, -0.3, 0.3, 0.3]), "shoulders/elbows follow +-0.3 rad", 0.1)
    track(arm_pub, arms, q0, "arms return", 0.1)

    print("== WBC walk: cmd_vel vx=-0.3 (backward, away from the table) for 4 sim-seconds ==")
    p0 = pelvis()
    twist = Twist()
    twist.linear.x = -0.3
    t_end = sim_now() + 4.0
    while sim_now() < t_end:
        cmd.publish(twist)
        spin(0.05)
    cmd.publish(Twist())
    spin(2.0)
    p1 = pelvis()
    moved = float(np.linalg.norm((p1 - p0)[:2]))
    upright = p1[2] > 0.6
    ok &= moved > 0.5 and upright
    print(f"  [{'PASS' if moved > 0.5 else 'FAIL'}] pelvis moved {moved:.2f} m in xy (want > 0.5)")
    print(f"  [{'PASS' if upright else 'FAIL'}] pelvis z {p1[2]:.3f} m (upright)")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
