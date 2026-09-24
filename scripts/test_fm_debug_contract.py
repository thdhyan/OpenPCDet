#!/usr/bin/env python3
"""Small contract check for the Foundation Model debug UI.

Run from the repository root:
    python scripts/test_fm_debug_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gr00t_ws_server import ARM_JOINTS, _action_to_arm_trajectory  # noqa: E402


def main() -> None:
    assert len(ARM_JOINTS) == 14
    current = np.arange(14, dtype=np.float32) * 0.01
    action = {
        "left_arm": np.ones((1, 40, 7), dtype=np.float32) * 0.01,
        "right_arm": np.ones((1, 40, 7), dtype=np.float32) * -0.01,
    }
    trajectory = _action_to_arm_trajectory(action, current)
    assert trajectory.shape == (40, 14)
    assert np.allclose(trajectory[0, :7], current[:7] + 0.01)
    assert np.allclose(trajectory[-1, :7], current[:7] + 0.4)
    print("FM debug contract: OK")


if __name__ == "__main__":
    main()
