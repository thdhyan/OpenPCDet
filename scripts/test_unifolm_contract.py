#!/usr/bin/env python3
"""Small contract check for the optional UnifoLM WebSocket adapter."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unifolm_ws_server import STATE_DIM, _images_from_message, _state_from_message


def main() -> None:
    # 9-D left pose + 9-D right pose + 5 waist/base values.
    state = [0.0] * 23
    state[0:3] = [0.35, 0.0, 1.05]
    state[3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    state[9:12] = [0.35, 0.0, 1.05]
    state[12:18] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert len(state) == STATE_DIM
    assert np.allclose(_state_from_message({"state": state}), state)
    try:
        _state_from_message({"joints": {"name": ["a"], "position": [0.0]}})
    except ValueError as exc:
        assert "joint angles alone" in str(exc)
    else:
        raise AssertionError("joint-only input must not be silently accepted")

    from PIL import Image

    image = Image.new("RGB", (8, 8), (10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    images = _images_from_message({"rgb": base64.b64encode(buffer.getvalue()).decode()})
    assert len(images) == 4 and all(item.size == (224, 224) for item in images)
    assert np.isfinite(_state_from_message({"state": state})).all()
    print("UnifoLM contract: OK")


if __name__ == "__main__":
    main()
