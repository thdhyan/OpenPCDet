#!/usr/bin/env python3
"""Lazy WebSocket adapter for Unitree UnifoLM-VLA.

The checkpoint is a 23-D end-effector policy: each pose is 9-D (XYZ plus a 6-D
rotation), followed by five waist/base values.  It is not a joint-action policy.
This adapter deliberately keeps EEF actions preview-only: joint states may be
logged, but they are not silently converted to EEF poses or dispatched to the
robot.

The official Unitree implementation is imported only on the first valid
observation so the UI/server can start while the large optional model is cold.

Run on Spark02:
  PYTHONPATH=~/UnifoLM-VLA/src ~/venvs/gr00t/bin/python \
    scripts/unifolm_ws_server.py \
    --ckpt ~/foundation_models/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt \
    --vlm ~/foundation_models/UnifoLM-VLM-Base --port 8767
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np


def _force_sdpa_attention() -> None:
    """Make the Unitree Qwen wrapper usable on CUDA 13 aarch64.

    Unitree's released wrapper hard-codes ``flash_attention_2``.  The available
    Spark wheel is CUDA-12-linked, so SDPA is the safe default; this avoids a
    hard failure without modifying the external checkout.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration

    cls = Qwen2_5_VLForConditionalGeneration
    if getattr(cls, "_fm_sdpa_patched", False):
        return
    original = cls.from_pretrained

    def patched(*args, **kwargs):
        kwargs["attn_implementation"] = "sdpa"
        return original(*args, **kwargs)

    cls.from_pretrained = staticmethod(patched)
    cls._fm_sdpa_patched = True

try:
    import websockets
except ImportError as exc:  # pragma: no cover - deployment dependency
    raise SystemExit("Install the 'websockets' package") from exc

log = logging.getLogger("unifolm-ws")

STATE_DIM = 23
ACTION_DIM = 23
DEFAULT_STATS_KEY = "g1_stack_block"
DEFAULT_TASK = "hold the current pose"


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric vector") from exc
    if vector.size != size or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return vector


def _decode_image(encoded: str) -> "Image.Image":
    from PIL import Image

    try:
        raw = base64.b64decode(encoded, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError("rgb must be base64-encoded image data") from exc


def _resize_image(image: "Image.Image") -> "Image.Image":
    # Match the checkpoint's 224x224 VLM input without adding a new image lib.
    return image.resize((224, 224))


def _bounds_stats(stats: dict[str, Any], normalized: np.ndarray, *, action: bool) -> np.ndarray:
    """Undo the Unitree bounds normalization used by the G1 dataset."""
    from unifolm_vla.rlds_dataloader.constants import (
        ACTION_PROPRIO_NORMALIZATION_TYPE,
        NormalizationType,
    )

    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        low_key, high_key = "min", "max"
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        low_key, high_key = "q01", "q99"
    else:
        raise RuntimeError(f"Unsupported normalization: {ACTION_PROPRIO_NORMALIZATION_TYPE}")

    mask = np.asarray(stats.get("mask", np.ones_like(stats[low_key], dtype=bool)), dtype=bool)
    low = np.asarray(stats[low_key], dtype=np.float32)
    high = np.asarray(stats[high_key], dtype=np.float32)
    if action:
        return np.where(mask, 0.5 * (normalized + 1.0) * (high - low + 1e-8) + low, normalized)
    return np.clip(np.where(mask, 2.0 * (normalized - low) / (high - low + 1e-8) - 1.0, normalized), -1.0, 1.0)


def _state_from_message(message: dict[str, Any]) -> np.ndarray:
    """Read the explicit 23-D EEF state contract.

    Layout: left XYZ + 6-D rotation (9), right XYZ + 6-D rotation (9), then five
    waist/base values. Joint names/positions are accepted for logging, but are
    never guessed into an EEF pose. A TF/IK producer must provide ``state``
    (23 values) or ``eef_state`` in exactly this Unitree layout.
    """
    value = message.get("state", message.get("eef_state"))
    if value is None:
        raise ValueError(
            "UnifoLM-VLA requires state/eef_state[23] "
            "(left XYZ+6D [9], right XYZ+6D [9], waist/base[5]); "
            "joint angles alone are not sufficient"
        )
    return _finite_vector(value, STATE_DIM, "state/eef_state")


def _images_from_message(message: dict[str, Any]) -> list["Image.Image"]:
    encoded = message.get("rgb")
    if not encoded:
        raise ValueError("obs requires an rgb image")
    full = _resize_image(_decode_image(encoded))
    images = message.get("images") or {}
    result = []
    # The training stack uses two wide/top and two wrist views.  The simulator
    # currently exposes one RGB camera, so use it as an explicit fallback.
    for key in ("left_top", "right_top", "left_wrist", "right_wrist"):
        item = images.get(key)
        result.append(_resize_image(_decode_image(item)) if item else full.copy())
    return result


class UnifoLMServer:
    def __init__(self, ckpt: str, vlm: str, stats_key: str, device: str,
                 use_bf16: bool, attention: str = "sdpa"):
        self.ckpt = ckpt
        self.vlm = vlm
        self.stats_key = stats_key
        self.device = device
        self.use_bf16 = use_bf16
        self.attention = attention
        self.model = None
        self.processor = None
        self.action_stats = None
        self.proprio_stats = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return
        if self.device.startswith("cuda") and not __import__("torch").cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        import torch
        from unifolm_vla.model.framework.base_framework import baseframework

        if self.attention == "sdpa":
            _force_sdpa_attention()
        log.info("loading UnifoLM checkpoint %s on %s (attention=%s)", self.ckpt, self.device, self.attention)
        model = baseframework.from_pretrained(self.ckpt, vlm_pretrained_path=self.vlm)
        if self.use_bf16:
            model = model.to(dtype=torch.bfloat16)
        model = model.to(self.device).eval()
        stats = model.norm_stats[self.stats_key]
        self.model = model
        self.processor = model.qwen_vl_interface.processor
        self.action_stats = stats["action"]
        self.proprio_stats = stats["proprio"]
        log.info("UnifoLM ready: %d-D state/action, stats=%s", ACTION_DIM, self.stats_key)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self.action_stats = None
        self.proprio_stats = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("UnifoLM unloaded")

    def infer(self, message: dict[str, Any]) -> dict[str, Any]:
        state = _state_from_message(message)
        images = _images_from_message(message)
        self.load()

        import torch
        from qwen_vl_utils import process_vision_info

        # The official model consumes the EEF state, not the simulator's 43
        # joint state.  Keep the normalization here explicit and inspectable.
        norm_state = _bounds_stats(self.proprio_stats, state, action=False)

        instruction = str(message.get("cmd", DEFAULT_TASK)).strip() or DEFAULT_TASK
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": f'The task is "{instruction.lower()}".'})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=prompt,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs["state"] = torch.from_numpy(norm_state).unsqueeze(0).to(self.device)
        for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            if key in inputs:
                inputs[key] = inputs[key].to(self.device)

        result = self.model.predict_action(qwen_inputs=inputs)
        normalized = np.asarray(result["normalized_actions"][0], dtype=np.float32)
        if normalized.ndim == 1:
            normalized = normalized[None, :]
        actions = _bounds_stats(self.action_stats, normalized, action=True).astype(np.float32)
        return {
            "type": "eef_action",
            "action_space": "eef23",
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "position": actions[0].tolist(),
            "trajectory": actions.tolist(),
            "joint_names": [],
            "preview_only": True,
            "requires_ik": True,
            "dispatch": "disabled_until_explicit_eef_to_ik",
            "task": instruction,
        }


async def _handle(ws, server: UnifoLMServer):
    peer = "unknown"
    try:
        peer = str(ws.transport.get_extra_info("peername"))
    except Exception:
        pass
    log.info("client connected: %s", peer)
    try:
        async for raw in ws:
            try:
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "load":
                    await asyncio.to_thread(server.load)
                    await ws.send(json.dumps({"type": "status", "loaded": True, "state_dim": STATE_DIM}))
                elif kind == "unload":
                    await asyncio.to_thread(server.unload)
                    await ws.send(json.dumps({"type": "status", "loaded": False}))
                elif kind == "obs":
                    response = await asyncio.to_thread(server.infer, message)
                    await ws.send(json.dumps(response))
                else:
                    await ws.send(json.dumps({"type": "status", "loaded": server.loaded}))
            except Exception as exc:
                log.exception("inference failed")
                await ws.send(json.dumps({"type": "error", "message": str(exc), "preview_only": True}))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        log.info("client disconnected: %s", peer)


async def _serve(args: argparse.Namespace):
    server = UnifoLMServer(
        args.ckpt, args.vlm, args.unnorm_key, args.device,
        not args.no_bf16, args.attention,
    )
    if args.load:
        server.load()
    async with websockets.serve(
        lambda ws: _handle(ws, server),
        args.host,
        args.port,
        max_size=50 * 1024 * 1024,
    ):
        log.info("UnifoLM WebSocket server listening on ws://%s:%d", args.host, args.port)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="UnifoLM-VLA pytorch_model.pt")
    parser.add_argument("--vlm", required=True, help="UnifoLM-VLM-Base directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "" else "cpu")
    parser.add_argument("--unnorm-key", default=DEFAULT_STATS_KEY)
    parser.add_argument("--attention", choices=("sdpa", "flash"), default="sdpa",
                        help="Qwen attention backend; SDPA is the Spark/CUDA13-safe default")
    parser.add_argument("--load", action="store_true", help="load at startup; default is lazy")
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
