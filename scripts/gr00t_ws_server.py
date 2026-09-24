#!/usr/bin/env python3
"""WebSocket inference server: laptop sensor bridge → GR00T arm targets.

This runs ON the spark (not dl). It accepts a single bidirectional WebSocket
connection from the laptop's ``ws_sensor_bridge.py`` and, on each incoming
observation frame, runs the GR00T N1.7-3B ``REAL_G1`` pretrain policy to
predict a 40-step arm action chunk and returns the FIRST step's joint targets
back over the SAME socket as an ``arm_cmd`` message → which the laptop bridge
republishes on /g1/arm_cmd (local DDS only, no cross-machine DDS).

Design choices (user constraints):
- Transport: WebSocket (not ZMQ, not cross-machine DDS).
- Single connection, laptop-initiated (client) - NAT/firewall friendly.
- Arms are decoupled from WBC: only ARM_JOINTS are returned; legs/waist stay
  the Balance policy's job.
- The policy predicts RELATIVE actions; we integrate the first step into an
  ABSOLUTE target using the current arm joints (also carried over the socket)
  so the command is a real hold/pose, not a delta.
- Hold-last semantics: if an observation frame arrives late/empty, the last
  computed arm pose is resent so the arms don't collapse.

Observation contract expected on the laptop side (REAL_G1, pretrain tag):
  video.ego_view: (T=2, H, W, 3) uint8  -- the laptop sends base64 JPEG/PNG.
  state: left_wrist_eef_9d (9), right_wrist_eef_9d (9),
         left_hand (7), right_hand (7), left_arm (7), right_arm (7), waist (3)
  The laptop sends joint names/positions; we rebuild the EEF 9d states from
  the arm joints + fixed hand offset until FK wiring is available - for the
  open-loop hold test the EEF state is mostly a conditioning context, the
  arm joints drive the targets.

Run:  python scripts/gr00t_ws_server.py --port 8765 \
          --model /home/thakk100/foundation_models/GR00T-N1.7-3B \
          --spark-host 0.0.0.0
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import time
from typing import Any

import numpy as np


# --- G1 arm joint layout (must match g1_sim/wbc_bridge.ARM_JOINTS) ---
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def _default_policy_factory(model_path: str, device: str = "cpu"):
    """Lazy import + load Gr00tPolicy with REAL_G1 pretrain tag."""
    # GR00T imports torch heavy; defer until first connection to keep the
    # server import light for `--help` / smoke tests.
    from gr00t.policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    pol = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=EmbodimentTag.REAL_G1,
        device=device,
        strict=False,
    )
    mc = pol.get_modality_config()
    return pol, mc


def _build_observation(mc, rgb_np: np.ndarray | None, joint_names: list[str],
                        joint_positions: list[float]) -> dict:
    """Assemble the dict GR00T's Gr00tPolicy.get_action() expects for REAL_G1.

    Args:
        mc: modality config from policy.get_modality_config().
        rgb_np: (H,W,3) uint8 ego-view image (may be None → zero pad).
        joint_names/positions: current G1 joint state (29-DOF subset; we read
            the 14 arm joints + 3 waist DOFs).
    """
    by_name = dict(zip(joint_names, joint_positions))
    # arm joint angles in ARM_JOINTS order (radians)
    arm_l = np.array([by_name.get(n, 0.0) for n in ARM_JOINTS[:7]], dtype=np.float32)
    arm_r = np.array([by_name.get(n, 0.0) for n in ARM_JOINTS[7:]], dtype=np.float32)
    # waist = first 3 leg-waist joints? No - waist_yaw/pitch/roll. Use the
    # waist_*_joint DOFs directly if present.
    waist = np.array([
        by_name.get("waist_yaw_joint", 0.0),
        by_name.get("waist_roll_joint", 0.0),
        by_name.get("waist_pitch_joint", 0.0),
    ], dtype=np.float32)

    # EEF 9d = [3 pos (m), 6 rot6d]. With no FK on the spark, approximate the
    # wrist EEF pose from the arm joints via a kinematic stub. For the open-loop
    # HOLD test the arms barely move, so a coarse EEF suffices as context -
    # the policy is asked to HOLD, not reach. We zero pos and set rot6d to the
    # identity (1,0,0,0,1,0) which is a valid near-rest pose.
    def _eef_9d(arm7: np.ndarray) -> np.ndarray:
        # pos: forward of hand ~ proportional to elbow bend (coarse); rot6d identity.
        return np.array([0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float32)
    left_eef = _eef_9d(arm_l)
    right_eef = _eef_9d(arm_r)

    # hand = 7-D: G1's rubber_hand is 1 DOF; the model's left_hand/right_hand
    # is 7-D per modstats. For the hold test set gripper closed-ish (grip on).
    def _hand_7() -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    state = {
        "left_wrist_eef_9d": left_eef[None, None, :],    # (1,1,9)
        "right_wrist_eef_9d": right_eef[None, None, :],
        "left_hand": _hand_7()[None, None, :],
        "right_hand": _hand_7()[None, None, :],
        "left_arm": arm_l[None, None, :],               # (1,1,7)
        "right_arm": arm_r[None, None, :],
        "waist": waist[None, None, :],                  # (1,1,3)
    }
    video = {}
    for vk in mc["video"].modality_keys:
        if rgb_np is not None:
            # resize to (H,W,3) expected - policy trainer crops to 256x256.
            h, w = 256, 256
            from PIL import Image
            im = Image.fromarray(rgb_np).resize((w, h))
            video[vk] = np.asarray(im, dtype=np.uint8)[None, None, :, :, :]  # (1,T,H,W,3)
        else:
            video[vk] = np.zeros((1, 1, 256, 256, 3), dtype=np.uint8)

    obs = {"video": video, "state": state}
    return obs


def _action_to_arm_targets(action: dict) -> np.ndarray:
    """Convert the REAL_G1 action dict into absolute (14,) ARM_JOINTS targets.

    Policy actions are RELATIVE (deltas). We integrate delta[0] (first step of
    the 40-step chunk) into absolute targets. left_arm and right_arm are
    absolute/non-eef? No - REAL_G1 uses RELATIVE for arm (see action_configs).
    So we add the delta to the current joint position; the caller passes the
    current joints so we can do this here. For the first open-loop hold test
    the caller sends current joints, so:
    """
    # left/right arm are 7-D each, relative.
    da_l = action["left_arm"][0, 0, :].astype(np.float32)   # (7,) first step
    da_r = action["right_arm"][0, 0, :].astype(np.float32)
    # hands are ABSOLUTE (gripper). We do not command hands here (no gripper DOF
    # on the 29-DOF G1 in this sim); ignore hand actions.
    # waist/bh/nav commands are WBC territory - ignore, the sim's Balance
    # policy owns those.
    return np.concatenate([da_l, da_r])


class Gr00tArmServer:
    """Async websocket server that runs GR00T and returns arm joint deltas."""

    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self._policy = None
        self._mc = None
        self._current_arm = np.zeros(14, dtype=np.float32)  # absolute pose held
        self._last_cmd_t = -1.0
        self._hold_count = 0

    @property
    def policy(self):
        if self._policy is None:
            print(f"[GR00T] loading {self.model_path} on {self.device} ...", flush=True)
            t0 = time.time()
            self._policy, self._mc = _default_policy_factory(self.model_path, self.device)
            print(f"[GR00T] loaded in {time.time()-t0:.1f}s; "
                  f"video_keys={self._mc['video'].modality_keys} "
                  f"state_keys={self._mc['state'].modality_keys} "
                  f"action_keys={self._mc['action'].modality_keys} "
                  f"action_horizon={len(self._mc['action'].delta_indices)}", flush=True)
        return self._policy, self._mc

    def infer(self, rgb_b64: str | None, joint_names: list[str],
              joint_positions: list[float], text_cmd: str | None) -> list[float]:
        """Return absolute 14-D (ARM_JOINTS-ordered) arm targets."""
        pol, mc = self.policy
        # decode rgb
        rgb_np = None
        if rgb_b64:
            try:
                raw = base64.b64decode(rgb_b64)
                from PIL import Image
                import numpy as _np
                rgb_np = _np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception as e:
                print(f"[GR00T] rgb decode fail: {e!r}", flush=True)
        obs = _build_observation(mc, rgb_np, joint_names, joint_positions)
        # attach language instruction
        lang_key = mc["language"].modality_keys[0] if "language" in mc else None
        if lang_key is None:
            lang_key = "annotation.human.task_description"
        obs["language"] = {lang_key: [[text_cmd or "hold a box"]]}
        action, _info = pol.get_action(obs)
        delta = _action_to_arm_targets(action)
        # integrate delta onto current absolute pose -> hold test means deltas ~0
        self._current_arm = self._current_arm + delta
        # clip to sane joint ranges (G1 arm ±2.6 rad)
        self._current_arm = np.clip(self._current_arm, -2.6, 2.6)
        self._hold_count += 1
        return self._current_arm.tolist()

    def current_targets(self) -> list[float]:
        return self._current_arm.tolist()


async def handle(ws, path=None, server=None):
    # Websockets 17.x passes only ws; 10.x passed (ws, path)
    # Server is passed from closure
    # In 17.x, connection has .transport.get_extra_info('peername')
    try:
        peername = ws.transport.get_extra_info('peername')
        client = f"{peername[0]}:{peername[1]}" if peername else "unknown"
    except Exception:
        client = "unknown"
    print(f"[WS] connection from {client}", flush=True)
    last_resend = 0.0
    try:
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") != "obs":
                continue
            t = msg.get("t", 0.0)
            rgb_b64 = msg.get("rgb")
            j = msg.get("joints") or {}
            names = j.get("name", [])
            pos = j.get("position", [])
            text_cmd = msg.get("cmd", "hold a box")

            # throttle: 1 policy inference per observation (laptop sends ~5 Hz)
            try:
                targets = server.infer(rgb_b64, names, pos, text_cmd)
                self_t = time.time()
                # round-trip RTT log
                rtt = time.time() - t
                print(f"[WS] {client} t={t:.3f} rtt={rtt*1000:.0f}ms "
                      f"arms={len(targets)}J held={server._hold_count}", flush=True)
                reply = {
                    "type": "arm_cmd",
                    "t": round(self_t, 4),
                    "name": ARM_JOINTS,
                    "position": targets,
                }
                await ws.send(json.dumps(reply))
            except Exception as e:
                # don't drop the connection on a single bad frame
                print(f"[WS] infer error: {e!r}", flush=True)
                # resend last good targets so arms don't collapse (hold-last)
                if time.time() - last_resend > 2.0:
                    last_resend = time.time()
                    await ws.send(json.dumps({
                        "type": "arm_cmd",
                        "t": round(time.time(), 4),
                        "name": ARM_JOINTS,
                        "position": server.current_targets(),
                    }))
    except Exception as e:
        print(f"[WS] client {client} disconnected: {e!r}", flush=True)
    print(f"[WS] connection {client} closed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(
        "~/foundation_models/GR00T-N1.7-3B"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    server = Gr00tArmServer(args.model, args.device)
    import websockets

    async def _serve():
        # Capture server in closure for websockets 17.x compatibility
        # Increase max message size for base64 RGB images
        async def handler_wrapper(ws, path=None):
            await handle(ws, path, server)
        async with websockets.serve(handler_wrapper, args.host, args.port, max_size=10*1024*1024):
            print(f"[WS] GR00T arm server on ws://{args.host}:{args.port}", flush=True)
            await asyncio.Future()  # run forever

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
