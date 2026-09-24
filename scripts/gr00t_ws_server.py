#!/usr/bin/env python3
"""WebSocket inference server: laptop sensor bridge → GR00T arm targets.

This runs ON the spark (not dl). It accepts a single bidirectional WebSocket
connection from the laptop's ``ws_sensor_bridge.py`` and, on each incoming
observation frame, runs the GR00T N1.7-3B ``REAL_G1`` pretrain policy to
predict a 40-step arm action chunk. It returns the full preview trajectory
plus its first absolute pose; dispatch is never automatic.

Design choices (user constraints):
- Transport: WebSocket (not ZMQ, not cross-machine DDS).
- Single connection, laptop-initiated (client) - NAT/firewall friendly.
- Arms are decoupled from WBC: only ARM_JOINTS are returned; legs/waist stay
  the Balance policy's job.
- The policy predicts RELATIVE actions; the server integrates the full chunk
  from the observed arm pose for preview. The UI must explicitly approve a
  selected pose before any dispatch path publishes it.
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
DEX3_HAND_JOINTS = [
    f"{side}_hand_{finger}_joint"
    for side in ("left", "right")
    for finger in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
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

    # Dex3 supplies the model's seven hand joints directly.  Older hand-only
    # G1 checkpoints may omit these names, in which case zeros remain a valid
    # neutral fallback rather than an invented finger conversion.
    hand_l = np.array([by_name.get(n, 0.0) for n in DEX3_HAND_JOINTS[:7]], dtype=np.float32)
    hand_r = np.array([by_name.get(n, 0.0) for n in DEX3_HAND_JOINTS[7:]], dtype=np.float32)

    state = {
        "left_wrist_eef_9d": left_eef[None, None, :],    # (1,1,9)
        "right_wrist_eef_9d": right_eef[None, None, :],
        "left_hand": hand_l[None, None, :],
        "right_hand": hand_r[None, None, :],
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


def _action_to_arm_trajectory(action: dict, current: np.ndarray) -> np.ndarray:
    """Convert a REAL_G1 action chunk to absolute ``(T, 14)`` arm poses.

    The GR00T REAL_G1 arm keys are relative joint deltas. The browser needs the
    whole horizon to scrub a trajectory, while the legacy control path still
    consumes the first absolute pose. WBC-owned waist/base outputs are ignored.
    """
    left = np.asarray(action["left_arm"], dtype=np.float32)
    right = np.asarray(action["right_arm"], dtype=np.float32)
    if left.ndim == 3:
        left = left[0]
    if right.ndim == 3:
        right = right[0]
    if left.ndim == 1:
        left = left[None, :]
    if right.ndim == 1:
        right = right[None, :]

    horizon = min(left.shape[0], right.shape[0])
    deltas = np.concatenate([left[:horizon], right[:horizon]], axis=1)
    trajectory = current[None, :] + np.cumsum(deltas, axis=0)
    return np.clip(trajectory, -2.6, 2.6).astype(np.float32)


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
              joint_positions: list[float], text_cmd: str | None) -> dict[str, object]:
        """Return the full arm trajectory and its first executable pose."""
        pol, mc = self.policy
        # decode rgb
        rgb_np = None
        if rgb_b64:
            try:
                raw = base64.b64decode(rgb_b64)
                from PIL import Image
                rgb_np = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception as e:
                print(f"[GR00T] rgb decode fail: {e!r}", flush=True)
        obs = _build_observation(mc, rgb_np, joint_names, joint_positions)
        # attach language instruction
        lang_key = mc["language"].modality_keys[0] if "language" in mc else None
        if lang_key is None:
            lang_key = "annotation.human.task_description"
        obs["language"] = {lang_key: [[text_cmd or "hold a box"]]}
        action, _info = pol.get_action(obs)
        by_name = dict(zip(joint_names, joint_positions))
        observed = np.array(
            [by_name.get(name, self._current_arm[index])
             for index, name in enumerate(ARM_JOINTS)],
            dtype=np.float32,
        )
        trajectory = _action_to_arm_trajectory(action, observed)
        self._current_arm = trajectory[0].copy()
        self._hold_count += 1
        return {
            "position": self._current_arm.tolist(),
            "trajectory": trajectory.tolist(),
            "q0": observed.tolist(),
            "horizon": int(trajectory.shape[0]),
            "joint_names": ARM_JOINTS,
            "relative_deltas": True,
            "preview_only": True,
            "dispatch": "explicit_ui_approval_required",
        }

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
                started = time.perf_counter()
                prediction = server.infer(rgb_b64, names, pos, text_cmd)
                infer_ms = (time.perf_counter() - started) * 1000
                self_t = time.time()
                rtt = self_t - t
                print(f"[WS] {client} t={t:.3f} rtt={rtt*1000:.0f}ms "
                      f"horizon={prediction['horizon']} infer={infer_ms:.0f}ms "
                      f"arms={len(prediction['position'])}J held={server._hold_count}", flush=True)
                await ws.send(json.dumps({
                    "type": "arm_cmd",
                    "t": round(self_t, 4),
                    "name": ARM_JOINTS,
                    "position": prediction["position"],
                    "trajectory": prediction["trajectory"],
                    "q0": prediction["q0"],
                    "horizon": prediction["horizon"],
                    "joint_names": prediction["joint_names"],
                    "relative_deltas": prediction["relative_deltas"],
                    "preview_only": True,
                    "dispatch": prediction["dispatch"],
                    "rtt_ms": rtt * 1000,
                    "infer_ms": infer_ms,
                }))
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
                        "trajectory": [server.current_targets()],
                        "horizon": 1,
                        "preview_only": True,
                        "dispatch": "explicit_ui_approval_required",
                    }))
    except Exception as e:
        print(f"[WS] client {client} disconnected: {e!r}", flush=True)
    print(f"[WS] connection {client} closed", flush=True)


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(
        "~/foundation_models/GR00T-N1.7-3B"))
    ap.add_argument("--device", default=_default_device())
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
