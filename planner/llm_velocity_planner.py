#!/usr/bin/env python3
"""LLM velocity planner for env 6 (``--env social_static``).

    /sim/humans           (names + world poses, read from USD by the sim)
    /g1/detections/livox  (VoxelNeXt pedestrians, pelvis frame)
    /tf  World -> pelvis
        -> every --cadence s: robot-frame observation of the NAMED detections
        -> intent-sim LLMClient (gpt-6-luna, Responses API, function calling)
        -> set_velocity(vx, vy, wz, duration_s) held on /g1/cmd_vel at 20 Hz
           (the in-sim WBC walks it); interact()/listen() end the episode.

Names: each detection is matched to the nearest roster person within
NAME_MATCH_RADIUS (oracle identity - detection itself is still the LiDAR's).
The roster positions never reach the model; they are used for that match and
for scoring (success = every target within range at the terminal call).

Writes <out>/trace.json after every decision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray

sys.path.insert(0, os.environ.get("INTENT_SIM_DIR", "/intent-sim"))
from llm.llm_client import LLMClient  # noqa: E402  (loads intent-sim/.env keys)

LIMITS = {"vx": (-0.3, 0.6), "vy": (-0.3, 0.3), "wz": (-0.8, 0.8)}  # m/s, m/s, rad/s
INTERACT_RANGE = 1.2  # m, intent-sim config.py
LISTEN_RANGE = 2.0    # m
NAME_MATCH_RADIUS = 1.0  # m, detection <-> roster
WORLD, BASE = "World", "pelvis"

TOOLS = [
    {"type": "function", "function": {
        "name": "set_velocity",
        "description": "Walk with a body-frame velocity: vx forward (m/s), vy left (m/s), wz counter-clockwise "
                       "yaw rate (rad/s). Held for duration_s, then the robot stops unless you send a new command. "
                       f"Limits: vx {LIMITS['vx']}, vy {LIMITS['vy']}, wz {LIMITS['wz']}.",
        "parameters": {"type": "object", "properties": {
            "vx": {"type": "number"}, "vy": {"type": "number"}, "wz": {"type": "number"},
            "duration_s": {"type": "number", "description": "seconds to hold this command"},
        }, "required": ["vx", "vy", "wz", "duration_s"]},
    }},
    {"type": "function", "function": {
        "name": "interact",
        "description": f"Stop and interact with the named people. Ends the episode; succeeds only if ALL are within {INTERACT_RANGE} m.",
        "parameters": {"type": "object", "properties": {
            "target_names": {"type": "array", "items": {"type": "string"}},
            "action": {"type": "string", "enum": ["talk", "low_wave", "shake_hand", "stand_still"]},
        }, "required": ["target_names", "action"]},
    }},
    {"type": "function", "function": {
        "name": "listen",
        "description": f"Stop and listen to the named people. Ends the episode; succeeds only if ALL are within {LISTEN_RANGE} m.",
        "parameters": {"type": "object", "properties": {
            "target_names": {"type": "array", "items": {"type": "string"}},
        }, "required": ["target_names"]},
    }},
]
TERMINAL = {"interact": INTERACT_RANGE, "listen": LISTEN_RANGE}

SYSTEM_PROMPT = """You control a Unitree G1 humanoid robot walking in a warehouse. A whole-body controller turns your velocity commands into footsteps.

All positions are in the ROBOT frame, in meters: x = forward, y = left. Bearing 0 deg = straight ahead, +90 deg = to the robot's left.
Every {cadence:.0f} s you receive a fresh observation from the robot's LiDAR person detector and choose the next command.

People in this scene (names known in advance): {names}.
Detections are labelled with a name when they can be matched to one of these people, otherwise "unknown".

## Robot API
- set_velocity(vx, vy, wz, duration_s): walk with this body-frame velocity for duration_s seconds, then stop. Limits: vx {vx}, vy {vy}, wz {wz}.
- interact(target_names, action): ends the episode; succeeds only if ALL named people are within {interact:.1f} m.
- listen(target_names): ends the episode; succeeds only if ALL named people are within {listen:.1f} m.

Guidance: turn (wz) so the target is roughly ahead before walking forward - forward walking is the fastest and most stable. Keep at least 0.5 m between your body and any person, and do not walk between two people who are talking to each other. Call interact()/listen() only once you are close enough. Choose exactly one action per observation.

## Task
{task}"""


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _clamp(v: float, lo_hi) -> float:
    return max(lo_hi[0], min(lo_hi[1], float(v)))


class Planner(Node):
    def __init__(self, a):
        super().__init__("g1_llm_velocity_planner", parameter_overrides=[
            rclpy.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        self.a = a
        self.llm = LLMClient(provider=a.provider, model=a.model)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.cmd_pub = self.create_publisher(Twist, "/g1/cmd_vel", 10)
        self.create_subscription(String, "/sim/humans", self._on_humans,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Detection3DArray, a.detections, self._on_dets, 10)
        self.create_timer(0.05, self._tick)

        self.lock = threading.Lock()
        self.roster: dict[str, tuple[float, float]] = {}  # name -> world (x, y)
        self.dets: list[dict] = []  # latest detections, world frame
        self.dets_t = None
        self.last_seen: dict[str, tuple[float, float, float]] = {}  # name -> (t, wx, wy)
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_until = 0.0
        self.done = False
        self.t0 = None
        self.min_clearance: dict[str, float] = {}
        self.path_len, self.last_xy = 0.0, None
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.trace = {"task": a.task, "model": a.model, "provider": a.provider,
                      "cadence_s": a.cadence, "decisions": [], "result": None}

    # ── inputs ──────────────────────────────────────────────────────────────
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def robot_pose(self):
        """(x, y, yaw) of the pelvis in World, or None before TF arrives."""
        try:
            t = self.tf.lookup_transform(WORLD, BASE, Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y, _yaw(t.transform.rotation)

    def _on_humans(self, msg: String) -> None:
        with self.lock:
            self.roster = {h["name"]: (h["x"], h["y"]) for h in json.loads(msg.data)["humans"]}

    def _on_dets(self, msg: Detection3DArray) -> None:
        pose = self.robot_pose()
        if pose is None:
            return
        rx, ry, ryaw = pose
        c, s = math.cos(ryaw), math.sin(ryaw)
        dets = []
        for d in msg.detections:
            p = d.bbox.center.position  # pelvis frame
            score = d.results[0].hypothesis.score if d.results else 0.0
            dets.append({"wx": rx + c * p.x - s * p.y, "wy": ry + s * p.x + c * p.y, "score": float(score)})
        with self.lock:
            # greedy one-to-one match, closest pairs first
            pairs = sorted(
                (math.hypot(d["wx"] - hx, d["wy"] - hy), i, n)
                for i, d in enumerate(dets) for n, (hx, hy) in self.roster.items()
            )
            used_d, used_n = set(), set()
            for dist, i, n in pairs:
                if dist <= NAME_MATCH_RADIUS and i not in used_d and n not in used_n:
                    dets[i]["name"] = n
                    used_d.add(i)
                    used_n.add(n)
                    self.last_seen[n] = (self.now(), dets[i]["wx"], dets[i]["wy"])
            self.dets, self.dets_t = dets, self.now()

    # ── 20 Hz: hold the command, score clearance ───────────────────────────
    def _tick(self) -> None:
        vx, vy, wz = self.cmd if (not self.done and self.now() < self.cmd_until) else (0.0, 0.0, 0.0)
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
        self.cmd_pub.publish(msg)

        pose = self.robot_pose()
        if pose is None or self.t0 is None:
            return
        with self.lock:
            for n, (hx, hy) in self.roster.items():
                d = math.hypot(hx - pose[0], hy - pose[1])
                self.min_clearance[n] = min(d, self.min_clearance.get(n, d))
        if self.last_xy is not None:
            self.path_len += math.hypot(pose[0] - self.last_xy[0], pose[1] - self.last_xy[1])
        self.last_xy = pose[:2]

    # ── decision loop (own thread: the LLM call blocks for seconds) ────────
    def observation(self, pose) -> str:
        rx, ry, ryaw = pose
        c, s = math.cos(-ryaw), math.sin(-ryaw)

        def to_robot(wx, wy):
            dx, dy = wx - rx, wy - ry
            return c * dx - s * dy, s * dx + c * dy

        now = self.now()
        lines = [f"t = {now - self.t0:.1f} s. You have walked {self.path_len:.1f} m."]
        with self.lock:
            dets, age, seen = list(self.dets), now - (self.dets_t or now), dict(self.last_seen)
            names = list(self.roster)
        lines.append(f"LiDAR detections ({age:.1f} s old):" if dets else "LiDAR detections: none.")
        for d in sorted(dets, key=lambda d: d.get("name", "~")):
            x, y = to_robot(d["wx"], d["wy"])
            lines.append(f"- {d.get('name', 'unknown')}: x={x:+.2f} y={y:+.2f} -> {math.hypot(x, y):.2f} m "
                         f"at bearing {math.degrees(math.atan2(y, x)):+.0f} deg (score {d['score']:.2f})")
        detected = {d.get("name") for d in dets}
        for n in names:
            if n in detected:
                continue
            if n in seen:
                t, wx, wy = seen[n]
                x, y = to_robot(wx, wy)
                lines.append(f"- {n}: not detected now; last seen {now - t:.0f} s ago at x={x:+.2f} y={y:+.2f} "
                             f"({math.hypot(x, y):.2f} m, bearing {math.degrees(math.atan2(y, x)):+.0f} deg)")
            else:
                lines.append(f"- {n}: not detected yet")
        if self.trace["decisions"]:
            lines.append(f"Your previous action: {json.dumps(self.trace['decisions'][-1]['calls'])}")
        return "\n".join(lines)

    def run(self) -> None:
        # wait for TF, the roster and the detector, then let the WBC settle
        while rclpy.ok() and (self.robot_pose() is None or not self.roster or self.dets_t is None):
            self.get_logger().info("waiting for /tf World->pelvis, /sim/humans, detections ...",
                                   throttle_duration_sec=5.0)
            time.sleep(0.5)
        start = self.now() + self.a.settle
        while rclpy.ok() and self.now() < start:
            time.sleep(0.1)
        self.t0 = self.now()
        with self.lock:
            names = ", ".join(self.roster)
        system = SYSTEM_PROMPT.format(cadence=self.a.cadence, names=names, task=self.a.task,
                                      interact=INTERACT_RANGE, listen=LISTEN_RANGE, **LIMITS)
        self.trace["system_prompt"] = system
        self.get_logger().info(f"episode start - roster: {names}; task: {self.a.task}")

        while rclpy.ok() and not self.done:
            t_obs = self.now()
            if t_obs - self.t0 > self.a.timeout:
                self.finish(None, "timeout")
                break
            pose = self.robot_pose()
            obs = self.observation(pose)
            wall = time.time()
            try:
                resp = self.llm.get_robot_policy(system, tools=TOOLS, max_rounds=1, user_turn=obs)
                calls, raw, err = resp.calls, resp.raw_content, None
            except Exception as e:  # keep the robot stopped, try again next cadence
                calls, raw, err = [], None, repr(e)
            latency = time.time() - wall
            self.trace["decisions"].append({"t": round(t_obs - self.t0, 2), "latency_s": round(latency, 2),
                                            "robot": [round(v, 3) for v in pose], "observation": obs,
                                            "calls": calls, "text": raw, "error": err})
            self.get_logger().info(f"[{t_obs - self.t0:6.1f}s] {latency:.1f}s LLM -> {calls or raw or err}")
            for call in calls:
                args = call.get("args", {})
                if call["function"] == "set_velocity":
                    self.cmd = (_clamp(args.get("vx", 0), LIMITS["vx"]), _clamp(args.get("vy", 0), LIMITS["vy"]),
                                _clamp(args.get("wz", 0), LIMITS["wz"]))
                    self.cmd_until = self.now() + max(0.0, min(float(args.get("duration_s", self.a.cadence)),
                                                               self.a.cadence + 1.0))
                elif call["function"] in TERMINAL:
                    self.finish(call, "terminal")
                    break
            self.dump()
            while rclpy.ok() and not self.done and self.now() < t_obs + self.a.cadence:
                time.sleep(0.05)
        self.dump()

    def finish(self, call, reason: str) -> None:
        self.done, self.cmd = True, (0.0, 0.0, 0.0)
        pose = self.robot_pose()
        result = {"reason": reason, "t": round(self.now() - self.t0, 2), "path_len_m": round(self.path_len, 2),
                  "min_clearance_m": {n: round(d, 2) for n, d in self.min_clearance.items()},
                  "success": False}
        if call is not None:
            rng = TERMINAL[call["function"]]
            targets = call["args"].get("target_names", [])
            with self.lock:
                dist = {n: round(math.hypot(self.roster[n][0] - pose[0], self.roster[n][1] - pose[1]), 2)
                        for n in targets if n in self.roster}
            result.update(call=call, target_dist_m=dist,
                          success=bool(targets) and len(dist) == len(targets) and all(d <= rng for d in dist.values()))
        self.trace["result"] = result
        self.get_logger().info(f"episode end: {result}")

    def dump(self) -> None:
        (self.out / "trace.json").write_text(json.dumps(self.trace, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="Go talk to Dhyan.")
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--cadence", type=float, default=3.0, help="seconds (sim time) between LLM decisions")
    p.add_argument("--timeout", type=float, default=180.0, help="episode length, seconds (sim time)")
    p.add_argument("--settle", type=float, default=5.0, help="seconds to stand still before the first decision")
    p.add_argument("--detections", default="/g1/detections/livox")
    p.add_argument("--out", default=f"logs/eval/{time.strftime('%Y%m%d_%H%M%S')}")
    a = p.parse_args()

    rclpy.init()
    node = Planner(a)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        node.run()
        time.sleep(0.5)  # let _tick publish the final zero command
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.dump()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
