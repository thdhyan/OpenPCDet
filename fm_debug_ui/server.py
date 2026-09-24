#!/usr/bin/env python3
"""Foundation Model Debug UI — backend server.

Serves the HTML UI and provides:
- /api/robot/<robot_id>  — URDF metadata
- /api/models            — list of available models
- /ws/model              — WebSocket proxy to model server
- /ws/sim                — WebSocket proxy to Isaac Sim bridge
- /api/ik                — IK solver endpoint (SE3 → joint angles)
- /api/schema            — output schema definitions

Run: python server.py --port 8080
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

try:
    from aiohttp import web, WSMsgType
except ImportError:
    print("pip install aiohttp websockets numpy")
    raise

import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fm-ui")

REPO = Path(__file__).resolve().parent.parent
ASSET_DIR = Path(__file__).resolve().parent / "assets"

# ============================================================
# MODEL REGISTRY
# ============================================================
MODELS = {
    "gr00t": {
        "name": "GR00T N1.7-3B",
        "type": "action",
        "ws_url": "ws://localhost:8765",
        "description": "VLA policy — outputs joint actions",
        "schemas": ["action-upper-body", "action-full-body", "joint-angles", "eef-se3"],
    },
    "cosmos-edge": {
        "name": "Cosmos3-Edge",
        "type": "video",
        "ws_url": "ws://localhost:8770",
        "description": "World model — next video prediction",
        "schemas": ["next-image", "next-video"],
    },
    "cosmos-nano": {
        "name": "Cosmos3-Nano",
        "type": "video",
        "ws_url": "ws://localhost:8771",
        "description": "World model — lightweight",
        "schemas": ["next-image", "next-video"],
    },
    "unifolm-vla": {
        "name": "UnifoLM-VLA",
        "type": "action",
        "ws_url": "ws://localhost:8767",
        "description": "Unitree G1 policy — EEF pose and base action chunks",
        "schemas": ["eef-se3", "action-upper-body"],
        "state_dim": 23,
        "requires_ik": True,
    },
}

# ============================================================
# URDF METADATA (G1 29-DOF)
# ============================================================
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LEG_WAIST_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]
ALL_JOINTS = LEG_WAIST_JOINTS + ARM_JOINTS

ROBOTS = {
    "g1-29dof": {
        "name": "Unitree G1 29-DOF",
        "num_joints": 29,
        "total_dof": 29,
        "arm_joints": ARM_JOINTS,
        "leg_waist_joints": LEG_WAIST_JOINTS,
        "urdf_url": "/assets/g1_29dof.urdf",
        "urdf_path": str(ASSET_DIR / "g1_29dof.urdf"),
    },
}

# ============================================================
# SCHEMAS
# ============================================================
SCHEMAS = {
    "action-upper-body": {
        "name": "Upper Body Action",
        "fields": [
            {"name": "left_arm", "type": "float32[7]", "desc": "Left arm joint angles (rad)"},
            {"name": "right_arm", "type": "float32[7]", "desc": "Right arm joint angles (rad)"},
            {"name": "left_hand", "type": "float32[7]", "desc": "Left hand/gripper"},
            {"name": "right_hand", "type": "float32[7]", "desc": "Right hand/gripper"},
        ]
    },
    "action-full-body": {
        "name": "Full Body Action",
        "fields": [
            {"name": "left_arm", "type": "float32[7]"}, {"name": "right_arm", "type": "float32[7]"},
            {"name": "left_leg", "type": "float32[6]"}, {"name": "right_leg", "type": "float32[6]"},
            {"name": "waist", "type": "float32[3]"}, {"name": "base_height", "type": "float32[1]"},
            {"name": "navigate", "type": "float32[3]"},
        ]
    },
    "next-image": {
        "name": "Next Image Prediction",
        "fields": [
            {"name": "image", "type": "uint8[H,W,3]"}, {"name": "depth", "type": "float32[H,W]"},
        ]
    },
    "next-video": {
        "name": "Next Video Prediction",
        "fields": [{"name": "frames", "type": "uint8[T,H,W,3]"}, {"name": "fps", "type": "int"}],
    },
    "eef-se3": {
        "name": "EEF SE(3) Poses",
        "fields": [
            {"name": "left_eef_pos", "type": "float32[3]"}, {"name": "left_eef_rot", "type": "float32[3x3]"},
            {"name": "right_eef_pos", "type": "float32[3]"}, {"name": "right_eef_rot", "type": "float32[3x3]"},
        ]
    },
    "joint-angles": {
        "name": "Joint Angle Values",
        "fields": [{"name": j, "type": "float32", "desc": "rad"} for j in ALL_JOINTS],
    },
    "language": {
        "name": "Language Output",
        "fields": [{"name": "text", "type": "string"}],
    },
}


# ============================================================
# SIMPLE IK SOLVER (differential IK for G1 arms)
# ============================================================
class SimpleIKSolver:
    """Numerical IK: given EEF target SE(3), compute joint angles via
    damped least-squares on a simple kinematic chain model.

    For real G1, this should be replaced with CuRobo or Pinocchio.
    This is a placeholder that does approximate forward kinematics.
    """

    # G1 arm link lengths (m)
    LINK_LENGTHS = [0.1, 0.1, 0.1, 0.1, 0.06, 0.04]  # shoulder->elbow chain
    BASE_OFFSET = np.array([0.0, 0.0, 0.0])  # wrist offset from shoulder

    def solve(self, target_pos: np.ndarray, current_joints: np.ndarray,
              side: str = "left", max_iters: int = 50) -> np.ndarray:
        """Simple numerical IK. Returns 7 joint angles for one arm."""
        q = current_joints[:7].copy().astype(np.float64)

        for _ in range(max_iters):
            pos = self._fk(q)
            err = target_pos - pos
            if np.linalg.norm(err) < 1e-4:
                break
            J = self._jacobian(q)
            # Damped least-squares
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)
            q += np.clip(dq, -0.1, 0.1)
            q = np.clip(q, -2.6, 2.6)  # G1 joint limits

        return q

    def _fk(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics: joint angles -> EEF position."""
        pos = np.zeros(3)
        angle = 0.0
        for i, l in enumerate(self.LINK_LENGTHS):
            angle += q[i]
            pos[0] += l * np.cos(angle)  # forward
            pos[2] += l * np.sin(angle)  # up
        return pos

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        """Numerical Jacobian of FK."""
        eps = 1e-6
        J = np.zeros((3, 7))
        pos0 = self._fk(q)
        for i in range(7):
            dq = q.copy()
            dq[i] += eps
            pos1 = self._fk(dq)
            J[:, i] = (pos1 - pos0) / eps
        return J


# ============================================================
# HTTP HANDLERS
# ============================================================
async def handle_index(request: web.Request) -> web.Response:
    index = Path(__file__).parent / "index.html"
    return web.FileResponse(index)


async def handle_models(request: web.Request) -> web.Response:
    return web.json_response(MODELS)


async def handle_schema(request: web.Request) -> web.Response:
    return web.json_response(SCHEMAS)


async def handle_robot(request: web.Request) -> web.Response:
    robot_id = request.match_info.get("robot_id", "g1-29dof")
    if robot_id not in ROBOTS:
        return web.json_response({"error": f"unknown robot: {robot_id}"}, status=404)
    return web.json_response(ROBOTS[robot_id])


async def handle_ik(request: web.Request) -> web.Response:
    """POST /api/ik — solve IK for EEF target."""
    try:
        data = await request.json()
        target_pos = np.array(data["target_pos"], dtype=np.float64)
        current = np.array(data.get("current_joints", [0] * 7), dtype=np.float64)
        side = data.get("side", "left")
        solver = SimpleIKSolver()
        result = solver.solve(target_pos, current, side)
        return web.json_response({
            "success": True,
            "side": side,
            "joint_angles": result.tolist(),
            "converged_error": float(np.linalg.norm(target_pos - solver._fk(result))),
        })
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=400)


# ============================================================
# WEBSOCKET PROXY — model
# ============================================================
async def ws_model_proxy(request: web.Request) -> web.WebSocketResponse:
    """Proxy WebSocket between browser and model server."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    log.info("Browser connected to /ws/model")

    try:
        import websockets
    except ImportError:
        await ws.send_json({"type": "error", "message": "websockets not installed"})
        return ws

    # Get model from query param
    model_id = request.query.get("model", "gr00t")
    model_info = MODELS.get(model_id, MODELS["gr00t"])
    model_url = model_info["ws_url"]

    try:
        async with websockets.connect(model_url, max_size=10 * 1024 * 1024) as model_ws:
            log.info(f"Connected to model server: {model_url}")

            async def browser_to_model():
                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        await model_ws.send(msg.data)
                    elif msg.type == WSMsgType.BINARY:
                        await model_ws.send(msg.data)

            async def model_to_browser():
                async for msg in model_ws:
                    if isinstance(msg, bytes):
                        await ws.send_bytes(msg)
                    else:
                        await ws.send_str(msg)

            tasks = [asyncio.create_task(browser_to_model()),
                     asyncio.create_task(model_to_browser())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

    except Exception as e:
        log.error(f"Model proxy error: {e}")
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        log.info("Browser disconnected from /ws/model")

    return ws


# ============================================================
# WEBSOCKET PROXY — sim
# ============================================================
async def ws_sim_proxy(request: web.Request) -> web.WebSocketResponse:
    """Proxy WebSocket between browser and Isaac Sim bridge."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    log.info("Browser connected to /ws/sim")

    try:
        import websockets
    except ImportError:
        await ws.send_json({"type": "error", "message": "websockets not installed"})
        return ws

    sim_url = request.query.get("sim_url", "ws://localhost:8766")

    try:
        async with websockets.connect(sim_url, max_size=50 * 1024 * 1024) as sim_ws:
            log.info(f"Connected to sim bridge: {sim_url}")

            async def browser_to_sim():
                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        await sim_ws.send(msg.data)
                    elif msg.type == WSMsgType.BINARY:
                        await sim_ws.send(msg.data)

            async def sim_to_browser():
                async for msg in sim_ws:
                    if isinstance(msg, bytes):
                        await ws.send_bytes(msg)
                    else:
                        await ws.send_str(msg)

            tasks = [asyncio.create_task(browser_to_sim()),
                     asyncio.create_task(sim_to_browser())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

    except Exception as e:
        log.error(f"Sim proxy error: {e}")
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        log.info("Browser disconnected from /ws/sim")

    return ws


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = web.Application()
    app.router.add_static("/assets", ASSET_DIR, show_index=False)
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/models", handle_models)
    app.router.add_get("/api/schema", handle_schema)
    app.router.add_get("/api/robot/{robot_id}", handle_robot)
    app.router.add_post("/api/ik", handle_ik)
    app.router.add_get("/ws/model", ws_model_proxy)
    app.router.add_get("/ws/sim", ws_sim_proxy)

    log.info(f"Starting UI server on http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
