"""External arm-target override channel for the G1's 14 non-WBC arm joints.

The decoupled_wbc Balance/Walk policies only command LEG_WAIST_JOINTS; the
arms are held by a fixed PD target set once at startup (see
``g1_warehouse_sim.py``). This module lets an outside ROS 2 publisher -
typically a foundation-model client (GR00T/Cosmos) on another machine -
stream new arm joint targets on ``/g1/arm_cmd`` without ever touching the
balance policy's joints: the warehouse loop applies whatever is fresh via
``set_joint_position_targets(..., joint_names=ARM_JOINTS)`` on the main
thread (Isaac Sim's articulation API is not thread-safe, so targets are
stashed by the subscription callback and consumed by the loop - the same
pattern as ``g1_sim/joint_cmd_sub.py`` for external WBC).

Message contract (``sensor_msgs/JointState``):

- ``name``/``position`` must be equal length; names are Unitree G1 joint
  names (a subset is fine - unnamed arm joints keep their last target).
- Isaac's PD keeps driving toward the last applied target, so a sender that
  stops publishing leaves the arms frozen at the last command - which for a
  "hold this pose" task is the desired semantics (chunks of a 40-step action
  horizon simply hold between re-plans).
- QoS is best-effort on the subscription side so both reliable and
  best-effort publishers are compatible.
"""

from __future__ import annotations

import time

import numpy as np

from g1_sim.wbc_bridge import ARM_JOINTS

ARM_CMD_TOPIC = "/g1/arm_cmd"
# Names accepted from publishers (arm joints only - leg/waist targets are
# the WBC's exclusive territory and are silently dropped if sent).
ARM_SET = frozenset(ARM_JOINTS)

# Dex3 fingers (7 per hand), same JointState contract on their own topic.
HAND_CMD_TOPIC = "/g1/hand_cmd"
DEX3_HAND_JOINTS = [
    f"{side}_hand_{finger}_joint"
    for side in ("left", "right")
    for finger in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
]


class ArmTargetSubscriber:
    """Buffers JointState targets for a fixed joint group (default: the 14
    arm joints on ``/g1/arm_cmd``; also used for the Dex3 fingers on
    ``/g1/hand_cmd``) for the warehouse loop."""

    def __init__(self, node, topic: str = ARM_CMD_TOPIC, joints: list[str] = ARM_JOINTS):
        from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import JointState

        self.joints = list(joints)
        self._index = {name: i for i, name in enumerate(self.joints)}
        self._targets = np.zeros(len(self.joints), dtype=np.float32)
        self._have_cmd = False
        self._last_rx = 0.0
        self._received = 0
        self._dropped = 0

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self._node = node
        self._sub = node.create_subscription(JointState, topic, self._callback, qos)

    def _callback(self, msg) -> None:
        positions = msg.position
        applied = 0
        for name, pos in zip(msg.name, positions):
            idx = self._index.get(name)
            if idx is None:
                self._dropped += 1
                continue
            self._targets[idx] = pos
            applied += 1
        if applied:
            self._have_cmd = True
            self._last_rx = time.monotonic()
            self._received += 1

    def spin_once(self) -> None:
        import rclpy

        rclpy.spin_once(self._node, timeout_sec=0.0)

    def get_targets(self) -> np.ndarray | None:
        """Latest targets in ``self.joints`` order, or None if never commanded."""
        return self._targets.copy() if self._have_cmd else None

    def age(self) -> float:
        """Seconds since the last command (inf if none yet)."""
        return time.monotonic() - self._last_rx if self._have_cmd else float("inf")

    @property
    def stats(self) -> tuple[int, int]:
        return self._received, self._dropped
