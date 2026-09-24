#!/usr/bin/env python3
"""Small contract check for the Foundation Model debug UI.

Run from the repository root:
    python scripts/test_fm_debug_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gr00t_ws_server import (  # noqa: E402
    ARM_JOINTS,
    DEX3_HAND_JOINTS,
    _action_to_arm_trajectory,
    _build_observation,
    Gr00tArmServer,
)


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

    modality = {"video": SimpleNamespace(modality_keys=["ego_view"])}
    joint_names = ARM_JOINTS + DEX3_HAND_JOINTS
    joint_positions = [0.0] * len(ARM_JOINTS) + [0.1] * len(DEX3_HAND_JOINTS)
    observation = _build_observation(modality, None, joint_names, joint_positions)
    assert np.allclose(observation["state"]["left_wrist_eef_9d"][0, 0, 3:].reshape(2, 3), np.eye(3)[:2])
    assert np.allclose(observation["state"]["left_hand"], 0.1)
    assert np.allclose(observation["state"]["right_hand"], 0.1)
    server = Gr00tArmServer("unused", "cpu")
    server.unload()
    assert server._policy is None
    print("FM debug contract: OK")


if __name__ == "__main__":
    main()
