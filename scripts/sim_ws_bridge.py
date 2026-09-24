#!/usr/bin/env python3
"""Isaac Sim WebSocket Bridge — streams camera, joints, and approved arm targets.

Runs INSIDE the Isaac Sim container that owns the ROS 2 graph. It subscribes to
``/g1/camera/rgb``, ``/g1/camera/depth``, and ``/g1/joint_states`` and forwards
them to the Foundation Model Debug UI. The UI may send an explicit ``arm_cmd``
message; only validated 14-joint arm targets are queued for publication on
``/g1/arm_cmd``. No model action is automatically forwarded.

Run (inside container):
  python scripts/sim_ws_bridge.py --port 8766
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import time
from typing import Any

import numpy as np

try:
    import websockets
except ImportError:
    print("pip install websockets")
    raise

log = logging.getLogger("sim-ws-bridge")

ARM_JOINTS = frozenset({
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
})
MAX_ARM_JOINT_RAD = 3.0


class SimBridge:
    """WebSocket server that streams sim data to connected UI clients."""

    def __init__(self, port: int = 8766, rate_hz: float = 10.0):
        self.port = port
        self.rate_hz = rate_hz
        self.clients: set = set()
        self.latest_rgb: str | None = None
        self.latest_depth: str | None = None
        self.latest_joints: dict | None = None
        self.latest_tf: list[dict] = []
        self.latest_tf_static: list[dict] = []
        self._pending_arm_command: tuple[list[str], list[float]] | None = None
        self._running = False

    async def start(self):
        async with websockets.serve(self._handle_client, "0.0.0.0", self.port, max_size=50 * 1024 * 1024):
            log.info(f"Sim WebSocket bridge on ws://0.0.0.0:{self.port}")
            self._running = True
            while self._running:
                await asyncio.sleep(1.0 / self.rate_hz)
                await self._broadcast_data()

    async def _handle_client(self, ws):
        client_id = id(ws)
        self.clients.add(ws)
        log.info(f"Client connected: {client_id} ({len(self.clients)} total)")

        try:
            async for msg in ws:
                data = json.loads(msg)
                if data.get("type") == "subscribe":
                    log.info(f"Client {client_id} subscribed: {data.get('topics', [])}")
                    await ws.send(json.dumps({
                        "type": "status",
                        "message": "subscribed",
                        "topics": data.get("topics", []),
                        "rate_hz": self.rate_hz,
                    }))
                elif data.get("type") == "arm_cmd":
                    names = data.get("name", [])
                    positions = data.get("position", [])
                    try:
                        finite_positions = [float(position) for position in positions]
                    except (TypeError, ValueError):
                        finite_positions = []
                    valid = (
                        names
                        and len(names) == len(positions)
                        and len(names) <= len(ARM_JOINTS)
                        and set(names) <= ARM_JOINTS
                        and len(finite_positions) == len(positions)
                        and all(np.isfinite(finite_positions))
                        and all(abs(position) <= MAX_ARM_JOINT_RAD for position in finite_positions)
                    )
                    if not valid:
                        await ws.send(json.dumps({
                            "type": "error",
                            "message": "arm_cmd rejected: expected finite arm joint targets within ±3.0 rad",
                        }))
                        continue
                    self._pending_arm_command = (list(names), [float(p) for p in positions])
                    log.info(f"Client {client_id} queued {len(names)} approved arm targets")
                    await ws.send(json.dumps({
                        "type": "status",
                        "message": "arm_cmd queued",
                        "joints": len(names),
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            log.info(f"Client disconnected: {client_id} ({len(self.clients)} remaining)")

    async def _broadcast_data(self):
        if not self.clients:
            return

        msg: dict[str, Any] = {"type": "sensor_frame", "t": time.time()}

        if self.latest_rgb:
            msg["rgb"] = self.latest_rgb
        if self.latest_depth:
            msg["depth"] = self.latest_depth
        if self.latest_joints:
            msg["joint_states"] = self.latest_joints
            msg["type"] = "joint_states"
        elif self.latest_rgb or self.latest_depth:
            msg["type"] = "camera"
        if self.latest_tf:
            msg["tf"] = self.latest_tf
        if self.latest_tf_static:
            msg["tf_static"] = self.latest_tf_static

        if len(msg) <= 2:
            return

        data = json.dumps(msg)
        dead = set()
        for ws in self.clients:
            try:
                await ws.send(data)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.clients.discard(ws)

    def update_rgb(self, jpeg_b64: str):
        self.latest_rgb = jpeg_b64

    def update_depth(self, jpeg_b64: str):
        self.latest_depth = jpeg_b64

    def update_joints(self, names: list, positions: list):
        self.latest_joints = {"name": names, "position": positions, "t": time.time()}

    def update_tf(self, transforms: list[dict], static: bool = False) -> None:
        if static:
            self.latest_tf_static = transforms
        else:
            self.latest_tf = transforms

    def take_pending_arm_command(self) -> tuple[list[str], list[float]] | None:
        command, self._pending_arm_command = self._pending_arm_command, None
        return command


async def ros_bridge_task(bridge: SimBridge):
    """Subscribe to ROS2 topics and feed the WebSocket bridge."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        from sensor_msgs.msg import Image, JointState
        from tf2_msgs.msg import TFMessage
    except ImportError:
        log.warning("ROS2 not available — running in test mode (no data)")
        while True:
            bridge.update_joints(
                [f"joint_{i}" for i in range(14)],
                list(np.random.randn(14) * 0.3)
            )
            await asyncio.sleep(0.1)
        return

    import threading

    rclpy.init()
    node = Node("sim_ws_bridge")

    qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
    )

    def encode_image(msg: Image) -> str:
        try:
            from PIL import Image as PILImage
            import io

            h = msg.height
            w = msg.width
            data = np.frombuffer(msg.data, dtype=np.uint8)

            if msg.encoding == "rgb8":
                arr = data.reshape(h, w, 3)
            elif msg.encoding == "bgr8":
                arr = data.reshape(h, w, 3)[:, :, ::-1]
            elif msg.encoding == "mono8":
                arr = data.reshape(h, w)
            elif msg.encoding == "8UC1":
                arr = data.reshape(h, w)
            elif msg.encoding == "8UC3":
                arr = data.reshape(h, w, 3)[:, :, ::-1]
            elif msg.encoding == "32FC1":
                arr = data.reshape(h, w)
                dmin, dmax = np.percentile(arr, [1, 99])
                if dmax > dmin:
                    arr = np.clip((arr - dmin) / (dmax - dmin) * 255, 0, 255).astype(np.uint8)
            else:
                return ""

            pil = PILImage.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            log.error(f"Image encode error: {e}")
            return ""

    def on_rgb(msg):
        jpeg = encode_image(msg)
        if jpeg:
            bridge.update_rgb(jpeg)

    def on_depth(msg):
        jpeg = encode_image(msg)
        if jpeg:
            bridge.update_depth(jpeg)

    def on_joints(msg):
        bridge.update_joints(list(msg.name), list(msg.position))

    def serialize_tf(msg):
        transforms = []
        for transform in msg.transforms:
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            transforms.append({
                "parent": transform.header.frame_id,
                "child": transform.child_frame_id,
                "translation": [translation.x, translation.y, translation.z],
                "rotation": [rotation.x, rotation.y, rotation.z, rotation.w],
                "stamp": [
                    transform.header.stamp.sec,
                    transform.header.stamp.nanosec,
                ],
            })
        return transforms

    def on_tf(msg):
        bridge.update_tf(serialize_tf(msg), static=False)

    def on_tf_static(msg):
        bridge.update_tf(serialize_tf(msg), static=True)

    qos_static = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=100,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )

    node.create_subscription(Image, "/g1/camera/rgb", on_rgb, qos)
    node.create_subscription(Image, "/g1/camera/depth", on_depth, qos)
    node.create_subscription(JointState, "/g1/joint_states", on_joints, qos)
    node.create_subscription(TFMessage, "/tf", on_tf, qos)
    node.create_subscription(TFMessage, "/tf_static", on_tf_static, qos_static)
    arm_publisher = node.create_publisher(JointState, "/g1/arm_cmd", qos)

    log.info("ROS2 subscriptions created — waiting for data")

    def spin_thread():
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

    t = threading.Thread(target=spin_thread, daemon=True)
    t.start()

    while True:
        pending = bridge.take_pending_arm_command()
        if pending is not None:
            names, positions = pending
            command = JointState()
            command.name = names
            command.position = positions
            arm_publisher.publish(command)
            log.info(f"Published approved arm targets to /g1/arm_cmd ({len(names)} joints)")
        await asyncio.sleep(0.01)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--test-mode", action="store_true", help="Run without ROS2, generate dummy data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    bridge = SimBridge(port=args.port, rate_hz=args.rate_hz)

    async def run():
        ros_task = asyncio.create_task(ros_bridge_task(bridge))
        await bridge.start()

    asyncio.run(run())


if __name__ == "__main__":
    main()
