"""External whole-body-control entry point for the Isaac Sim G1.

Mirrors the Gazebo Harmonic contract from g1_perception_ws
(GAZEBO_BALANCE_FIX.md): an external ROS node publishes ONE
``std_msgs/Float64`` position target per joint at ~50 Hz::

    /g1/joint/left_hip_pitch_joint     ... x29 (ALL_JOINTS order names)

``JointCommandSubscriber`` buffers the latest value per joint;
the sim loop calls :meth:`get_targets` every control tick and forwards the
vector to ``Articulation.set_joint_position_targets`` - the per-joint PD
gains baked via ``set_gains`` do the tracking, exactly like the in-sim
decoupled_wbc bridge. Joints that have not received a message yet hold the
URDF default pose (0.0) until the first target arrives.
"""

from __future__ import annotations

import numpy as np


class JointCommandSubscriber:
    """ROS2 listener for per-joint Float64 position targets."""

    def __init__(self, node, joint_names: list[str], topic_prefix: str = "/g1/joint"):
        self.node = node
        self.joint_names = list(joint_names)
        self._targets = np.zeros(len(self.joint_names), dtype=np.float32)
        self._received = [False] * len(self.joint_names)
        from std_msgs.msg import Float64

        self._subs = [
            node.create_subscription(
                Float64, f"{topic_prefix}/{name}",
                (lambda i: lambda msg: self._cb(i, msg))(i), 10,
            )
            for i, name in enumerate(self.joint_names)
        ]

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        """Service pending rclpy callbacks without blocking the sim loop."""
        import rclpy

        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def _cb(self, idx: int, msg) -> None:
        self._targets[idx] = float(msg.data)
        self._received[idx] = True

    @property
    def all_received(self) -> bool:
        return all(self._received)

    def get_targets(self) -> np.ndarray | None:
        """Latest (29,) target vector, or None until EVERY joint has one."""
        if not self.all_received:
            return None
        return self._targets.copy()
