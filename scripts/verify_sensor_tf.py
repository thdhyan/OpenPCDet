#!/usr/bin/env python3
"""Verify lidar + D435 point clouds land in the right place in the World frame.

Run against a live sim (ROS 2 env sourced):
    python3 scripts/verify_sensor_tf.py [--frames 10] [--json out.json]

Checks:
  * every cloud's frame_id resolves to World through /tf + /tf_static
  * Mid-360: sensor-frame elevation spans the datasheet FOV (-7..+52 deg),
    azimuth covers 360 deg, lowest world-Z returns sit on the floor
  * D435 depth/points: floor returns sit at world Z ~ 0 and the cloud
    overlaps the RGB-fused depth/color/points cloud (same geometry)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener

WORLD = "World"
LIDAR = "/livox/mid360/points/a"
DEPTH = "/g1/camera/depth/points"
COLOR = "/g1/camera/depth/color/points"
MID360_ELEV_DEG = (-7.0, 52.0)
FLOOR_BAND_M = 0.10


def quat_to_R(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def xyz(msg: PointCloud2) -> np.ndarray:
    pts = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    arr = np.stack([pts["x"], pts["y"], pts["z"]], axis=-1).astype(np.float64)
    return arr[np.isfinite(arr).all(axis=1)]


class Collector(Node):
    def __init__(self, n_frames: int):
        super().__init__("sensor_tf_verify")
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.n = n_frames
        self.msgs: dict[str, list[PointCloud2]] = {LIDAR: [], DEPTH: [], COLOR: []}
        for topic in self.msgs:
            self.create_subscription(PointCloud2, topic, self._cb(topic), qos_profile_sensor_data)

    def _cb(self, topic):
        def cb(msg):
            if len(self.msgs[topic]) < self.n:
                self.msgs[topic].append(msg)
        return cb

    def done(self) -> bool:
        return all(len(v) >= self.n for v in self.msgs.values())

    def to_world(self, msg: PointCloud2):
        # TF at the cloud's own stamp: the robot sways while balancing, so the
        # latest TF applied to an older cloud tilts far returns by decimetres.
        tf = self.buf.lookup_transform(
            WORLD, msg.header.frame_id, rclpy.time.Time.from_msg(msg.header.stamp), Duration(seconds=1.0)
        )
        R = quat_to_R(tf.transform.rotation)
        t = tf.transform.translation
        return R, np.array([t.x, t.y, t.z])


def summarize_lidar(node: Collector, msgs, sensor_offset: np.ndarray) -> dict:
    parts = []
    for m in msgs:
        try:
            R, T = node.to_world(m)
        except TransformException:
            continue
        p = xyz(m)
        parts.append((p - sensor_offset, p @ R.T + T))
    if not parts:
        raise TransformException(f"no stamped TF for {msgs[-1].header.frame_id}")
    sensor = np.concatenate([s for s, _ in parts])
    world = np.concatenate([w for _, w in parts])
    rng = np.linalg.norm(sensor, axis=1)
    keep = rng > 0.05
    sensor, world, rng = sensor[keep], world[keep], rng[keep]
    elev = np.degrees(np.arcsin(np.clip(sensor[:, 2] / rng, -1, 1)))
    azim = np.degrees(np.arctan2(sensor[:, 1], sensor[:, 0]))
    az_bins = np.histogram(azim, bins=36, range=(-180, 180))[0]
    floor = world[world[:, 2] < FLOOR_BAND_M]
    return {
        "frame_id": msgs[-1].header.frame_id,
        "scans_used": len(parts),
        "points": int(len(sensor)),
        "azimuth_10deg_bin_counts": az_bins.tolist(),
        "elevation_deg_min": float(elev.min()),
        "elevation_deg_max": float(elev.max()),
        "elevation_deg_p1_p99": [float(np.percentile(elev, 1)), float(np.percentile(elev, 99))],
        "azimuth_deg_min": float(azim.min()),
        "azimuth_deg_max": float(azim.max()),
        "azimuth_10deg_bins_filled": int((az_bins > 0).sum()),
        "range_m_min_max": [float(rng.min()), float(rng.max())],
        "world_z_min": float(world[:, 2].min()),
        "world_z_p1": float(np.percentile(world[:, 2], 1)),
        "world_z_max": float(world[:, 2].max()),
        "world_z_mean": float(world[:, 2].mean()),
        "floor_points": int(len(floor)),
        "floor_z_mean": float(floor[:, 2].mean()) if len(floor) else None,
        "floor_z_std": float(floor[:, 2].std()) if len(floor) else None,
    }


def summarize_cam(node: Collector, msgs) -> tuple[dict, np.ndarray]:
    msg = msgs[-1]
    pts = xyz(msg)
    R, T = node.to_world(msg)
    world = pts @ R.T + T
    floor = world[world[:, 2] < FLOOR_BAND_M]
    return {
        "frame_id": msg.header.frame_id,
        "points": int(len(pts)),
        "camera_origin_world_z": float(T[2]),
        "world_z_min": float(world[:, 2].min()),
        "world_z_p1": float(np.percentile(world[:, 2], 1)),
        "world_z_max": float(world[:, 2].max()),
        "world_z_mean": float(world[:, 2].mean()),
        "floor_fraction": float(len(floor) / max(len(world), 1)),
        "floor_z_mean": float(floor[:, 2].mean()) if len(floor) else None,
        "floor_z_std": float(floor[:, 2].std()) if len(floor) else None,
    }, world


def nn_dist(a: np.ndarray, b: np.ndarray, sample: int = 3000) -> dict:
    rng = np.random.default_rng(0)
    a = a[rng.choice(len(a), min(sample, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(4 * sample, len(b)), replace=False)]
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return {"median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--json")
    ap.add_argument(
        "--lidar-offset",
        default="0,0,-0.15",
        help="sensor prim translation in mid360_link (load_g1 lidar_translation); "
        "removed before computing elevation/azimuth",
    )
    args = ap.parse_args()
    lidar_offset = np.array([float(v) for v in args.lidar_offset.split(",")])

    rclpy.init()
    node = Collector(args.frames)
    end = time.time() + args.timeout
    while time.time() < end and not node.done():
        rclpy.spin_once(node, timeout_sec=0.1)
    settle = time.time() + 2.0
    while time.time() < settle:
        rclpy.spin_once(node, timeout_sec=0.1)

    report: dict = {}
    worlds = {}
    lidar_fn = lambda n, m: summarize_lidar(n, m, lidar_offset)  # noqa: E731
    for topic, fn in ((LIDAR, lidar_fn), (DEPTH, summarize_cam), (COLOR, summarize_cam)):
        msgs = node.msgs[topic]
        if not msgs:
            report[topic] = {"error": "no data"}
            continue
        try:
            out = fn(node, msgs)
        except TransformException as e:
            report[topic] = {"frame_id": msgs[-1].header.frame_id, "error": f"TF: {e}"}
            continue
        if isinstance(out, tuple):
            report[topic], worlds[topic] = out
        else:
            report[topic] = out

    if DEPTH in worlds and COLOR in worlds:
        report["depth_vs_color_nn"] = nn_dist(worlds[DEPTH], worlds[COLOR])

    checks = []
    lid = report.get(LIDAR, {})
    if "elevation_deg_min" in lid:
        lo, hi = lid["elevation_deg_p1_p99"]
        checks.append(("lidar elevation within -7..52 deg (+-2)", lo >= MID360_ELEV_DEG[0] - 2 and hi <= MID360_ELEV_DEG[1] + 2))
        checks.append(("lidar elevation spans FOV (>=45 deg)", hi - lo >= 45))
        checks.append(("lidar azimuth covers 360 (>=34/36 bins)", lid["azimuth_10deg_bins_filled"] >= 34))
        checks.append(("lidar floor at z~0 (|mean|<0.05)", lid["floor_z_mean"] is not None and abs(lid["floor_z_mean"]) < 0.05))
        checks.append(("lidar lowest returns near ground (p1 z < 0.10)", lid["world_z_p1"] < 0.10))
        checks.append(("lidar nothing below floor (min z > -0.10)", lid["world_z_min"] > -0.10))
    for topic in (DEPTH, COLOR):
        cam = report.get(topic, {})
        if "world_z_min" in cam:
            checks.append((f"{topic} floor at z~0 (|mean|<0.05)", cam["floor_z_mean"] is not None and abs(cam["floor_z_mean"]) < 0.05))
            checks.append((f"{topic} nothing below floor (min z > -0.10)", cam["world_z_min"] > -0.10))
        else:
            checks.append((f"{topic} received + TF resolved", False))
    if "depth_vs_color_nn" in report:
        checks.append(("depth/points overlaps depth/color/points (median NN < 5 cm)", report["depth_vs_color_nn"]["median_m"] < 0.05))
    report["checks"] = [{"name": n, "pass": bool(p)} for n, p in checks]

    print(json.dumps(report, indent=2))
    print("\nRESULT:", "PASS" if all(p for _, p in checks) else "FAIL")
    for n, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {n}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
