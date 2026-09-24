#!/usr/bin/env python3
"""WebSocket bridge between the laptop Isaac Sim (ROS2 DDS on localhost only)
and the spark foundation-model server.

Transport policy (user requirement): all laptop↔spark communication goes over
**WebSocket**, NOT cross-machine DDS. ROS2 topics stay local to the laptop.

Direction 1 (laptop → spark): sensor observations.
    /g1/camera/rgb  (sensor_msgs/Image)
    /g1/joint_states (sensor_msgs/JointState)
    /g1/arm_cmd echo (for round-trip RTT logging)
Direction 2 (spark → laptop): arm targets.
    JointState on /g1/arm_cmd  (consumed by ArmTargetSubscriber already wired
    into the warehouse loop)

A single WebSocket connection is used bidirectionally. Laptop = client
(initiates the TCP connection - NAT/firewall friendly; the spark runs a
plain ws:// server on a known port).

Frame format (JSON text frames):
    {"type":"obs", "t":123.4, "rgb":"base64-jpeg", "joints":{"n":[...],"p":[...]}, "cmd":"hold a box"}
    {"type":"arm_cmd", "t":123.5, "name":[...], "position":[...]}
    {"type":"ping", "t":123.6}
    {"type":"cmd", "text":"pick up the box"}   # spark -> laptop: text trigger

Run:  python scripts/ws_sensor_bridge.py --host <spark_ip> --port 8765
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
from typing import Any

import numpy as np
from PIL import Image
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import JointState, Image
from tf2_msgs.msg import TFMessage

# Re-use the arm-override message contract.
from g1_sim.wbc_bridge import ARM_JOINTS
from g1_sim.arm_override import ARM_CMD_TOPIC


class WsBridgeNode(Node):
    """ROS2 side: subscribes to sensors, publishes arm commands."""

    def __init__(self, arm_topic: str = ARM_CMD_TOPIC):
        super().__init__("g1_ws_bridge")
        self._last_rgb: Image | None = None
        self._last_joints: JointState | None = None
        self._last_tf: TFMessage | None = None
        self._last_cmd: str | None = None  # text command, if any

        qos_sub = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self._rgb_sub = self.create_subscription(
            Image, "/g1/camera/rgb", self._on_rgb, qos_sub)
        self._joint_sub = self.create_subscription(
            JointState, "/g1/joint_states", self._on_joints, qos_sub)
        self._tf_sub = self.create_subscription(
            TFMessage, "/tf", self._on_tf, qos_sub)

        qos_pub = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self._arm_pub = self.create_publisher(JointState, arm_topic, qos_pub)
        self.get_logger().info(f"ws_bridge: subscribing /g1/camera/rgb, /g1/joint_states; "
                               f"publishing arm cmds on {arm_topic}")

    def _on_rgb(self, msg: Image) -> None:
        self._last_rgb = msg

    def _on_joints(self, msg: JointState) -> None:
        self._last_joints = msg

    def _on_tf(self, msg: TFMessage) -> None:
        self._last_tf = msg

    def take_observations(self):
        """Return a JPEG RGB snapshot, joint dictionary, and sim time."""
        rgb = self._last_rgb
        self._last_rgb = None
        j = self._last_joints
        self._last_joints = None
        t = self.get_clock().now().nanoseconds / 1e9
        rgb_b64 = None
        if rgb is not None and rgb.data:
            # GR00T expects an encoded image, not ROS's raw RGB8 buffer.
            # Honor Image.step so row padding is stripped before encoding.
            row_bytes = int(rgb.step)
            raw = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, row_bytes)
            pixels = raw[:, : rgb.width * 3].reshape(rgb.height, rgb.width, 3)
            if rgb.encoding == "bgr8":
                pixels = pixels[:, :, ::-1]
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="JPEG", quality=80)
            rgb_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        jdict = None
        if j is not None:
            jdict = {"name": list(j.name), "position": list(j.position)}
        if self._last_tf is not None:
            jdict = jdict or {"name": [], "position": []}
            jdict["tf"] = [
                {
                    "parent": transform.header.frame_id,
                    "child": transform.child_frame_id,
                    "translation": [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ],
                    "rotation": [
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w,
                    ],
                }
                for transform in self._last_tf.transforms
            ]
            self._last_tf = None
        return rgb_b64, jdict, t

    def publish_arm(self, names: list[str], positions: list[float], t: float) -> None:
        """Push an arm target onto /g1/arm_cmd (consumed by ArmTargetSubscriber)."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(p) for p in positions]
        self._arm_pub.publish(msg)
        self.get_logger().info(f"ws_bridge: published /g1/arm_cmd "
                               f"({len(names)} joints) @ {t:.3f}")


async def ws_client(bridge: WsBridgeNode, host: str, port: int,
                     text_cmd: str | None, publish_hz: float,
                     send_to_sim: bool = False) -> None:
    import websockets
    from websockets.exceptions import ConnectionClosed

    uri = f"ws://{host}:{port}"
    period = 1.0 / publish_hz
    reconnect = 0
    while True:
        try:
            async with websockets.connect(uri, ping_interval=5, ping_timeout=10) as ws:
                bridge.get_logger().info(f"ws_bridge: connected to {uri}")
                if not send_to_sim:
                    bridge.get_logger().warning(
                        "arm dispatch disabled; received actions are preview-only (use --send-to-sim to opt in)"
                    )
                loop = asyncio.get_event_loop()
                while True:
                    # spin ROS2 + collect obs without blocking the loop
                    rclpy.spin_once(bridge, timeout_sec=0.0)
                    rgb_b64, jdict, t = bridge.take_observations()
                    # if a static text command was supplied, always forward it
                    outgoing: dict[str, Any] = {
                        "type": "obs",
                        "t": round(t, 4),
                        "rgb": rgb_b64,
                        "joints": jdict,
                    }
                    if text_cmd:
                        outgoing["cmd"] = text_cmd
                    await ws.send(json.dumps(outgoing))

                    # read any inbound arm command
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=period)
                        msg = json.loads(raw)
                    except asyncio.TimeoutError:
                        continue
                    except ConnectionClosed:
                        raise

                    if msg.get("type") == "arm_cmd":
                        names = msg.get("name", [])
                        pos = msg.get("position", [])
                        if send_to_sim and names and pos:
                            loop.call_soon_threadsafe(
                                lambda n=names, p=pos, st=t: bridge.publish_arm(n, p, st)
                            )
                    elif msg.get("type") == "pong":
                        pass
                    await asyncio.sleep(0.0)
        except (OSError, ConnectionClosed, websockets.InvalidURI) as e:
            reconnect = min(reconnect + 2, 30)
            bridge.get_logger().warn(f"ws_bridge: {uri} {e!r} - retry in {reconnect}s")
            await asyncio.sleep(reconnect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.131.140.170",
                    help="spark IP (default aim_spark02)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--text-cmd", default="hold a box",
                    help="static text instruction forwarded every frame")
    ap.add_argument("--rate", type=float, default=5.0,
                    help="publish observations Hz")
    ap.add_argument("--send-to-sim", action="store_true",
                    help="publish model arm targets to /g1/arm_cmd; default is preview-only")
    args = ap.parse_args()

    rclpy.init()
    bridge = WsBridgeNode()
    try:
        asyncio.run(ws_client(bridge, args.host, args.port,
                              args.text_cmd, args.rate, args.send_to_sim))
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
