"""Bridge from planar velocity commands to decoupled_wbc's Balance/Walk policies.

Ported from NVIDIA GR00T-WholeBodyControl's ``decoupled_wbc``
(``sim2mujoco/scripts/run_mujoco_gear_wbc.py``), whose observation/action
contract was verified against MuJoCo (see
``docs/decoupled_wbc_findings.md``) before this port: the two ONNX policies
(``GR00T-WholeBodyControl-Balance.onnx``, ``...-Walk.onnx``) take a 516-dim
observation (6-step history of an 86-dim single frame) and output a 15-dim
action covering the legs + waist. The 14 arm joints are not policy-controlled
and are simply held near their default pose by a separate, fixed PD target.

This module only implements the observation/inference math - it has no
Isaac Sim imports, so it is the same code exercised standalone in the
scratchpad MuJoCo harness. The caller is responsible for feeding it
joint-name-labelled state and applying the returned targets via
``Articulation.set_joint_position_targets(..., joint_names=...)``, which
sidesteps any mismatch between decoupled_wbc's joint order and whatever
order Isaac Sim assigned the USD's articulation internally.
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

# Lazy import — onnxruntime only needed when WbcBridge is instantiated,
# not when importing joint-name constants for external WBC mode.
ort = None

# Canonical decoupled_wbc / Unitree 29-DOF joint order - see
# decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.xml (defines qpos
# order) and decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml
# (WeakMotorJointIndex, independent confirmation of the same order). The
# policy was trained on this exact order.
LEG_WAIST_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# Full 29-DOF order the observation is built in (qpos/qvel of ALL joints go
# into the obs, even though only the first 15 are policy-actuated).
ALL_JOINTS = LEG_WAIST_JOINTS + ARM_JOINTS

NUM_ACTIONS = len(LEG_WAIST_JOINTS)  # 15
NUM_JOINTS = len(ALL_JOINTS)  # 29
SINGLE_OBS_DIM = 86
OBS_HISTORY_LEN = 6
NUM_OBS = SINGLE_OBS_DIM * OBS_HISTORY_LEN  # 516

# From g1_gear_wbc.yaml - only covers LEG_WAIST_JOINTS; the remaining 14 arm
# dof are implicitly zero (padded) in the observation's qj_scaled term.
DEFAULT_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     0.0, 0.0, 0.0],
    dtype=np.float32,
)
_PADDED_DEFAULT_ANGLES = np.concatenate([DEFAULT_ANGLES, np.zeros(NUM_JOINTS - NUM_ACTIONS, dtype=np.float32)])

# From g1_gear_wbc.yaml's kps/kds - the policy was trained assuming these
# exact per-joint PD gains. run_mujoco_gear_wbc.py applies them as explicit
# torque (`(target - q) * kp - qvel * kd`) every physics step; Isaac Sim's
# articulation instead wants them set once as the implicit drive's
# stiffness/damping (`Articulation.set_gains(..., joint_names=...)`) - the
# USD's own baked-in gains (from convert_g1_urdf_to_usd.py, uniform 100/10
# for every joint) do NOT match this and were observed to make the robot
# collapse within ~2s when driven by the policy (see findings doc).
KP = np.array([150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250], dtype=np.float32)
KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5], dtype=np.float32)
# Arms are held by a much softer fixed PD in run_mujoco_gear_wbc.py.
ARM_KP = 100.0
ARM_KD = 0.5

CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
ACTION_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ANG_VEL_SCALE = 0.5
DEFAULT_HEIGHT_CMD = 0.74
# Below this command norm the Balance policy runs instead of Walk - mirrors
# run_mujoco_gear_wbc.py's `np.linalg.norm(loco_cmd) <= 0.05` switch.
WALK_CMD_DEADBAND = 0.05


def quat_rotate_inverse(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector ``v`` into the body frame given by quat_wxyz.

    Same math as run_mujoco_gear_wbc.py's ``quat_rotate_inverse``. Exposed so
    callers can rotate a world-frame angular velocity reading into body frame
    before calling :meth:`WbcBridge.step` - MuJoCo's free-joint qvel is
    already body-frame, but Isaac Sim's articulation angular velocity is not,
    so that conversion has to happen on the Isaac Sim side.
    """
    w, x, y, z = quat_wxyz
    qc = np.array([w, -x, -y, -z], dtype=np.float64)
    return np.array(
        [
            v[0] * (qc[0] ** 2 + qc[1] ** 2 - qc[2] ** 2 - qc[3] ** 2)
            + v[1] * 2 * (qc[1] * qc[2] - qc[0] * qc[3])
            + v[2] * 2 * (qc[1] * qc[3] + qc[0] * qc[2]),
            v[0] * 2 * (qc[1] * qc[2] + qc[0] * qc[3])
            + v[1] * (qc[0] ** 2 - qc[1] ** 2 + qc[2] ** 2 - qc[3] ** 2)
            + v[2] * 2 * (qc[2] * qc[3] - qc[0] * qc[1]),
            v[0] * 2 * (qc[1] * qc[3] - qc[0] * qc[2])
            + v[1] * 2 * (qc[2] * qc[3] + qc[0] * qc[1])
            + v[2] * (qc[0] ** 2 - qc[1] ** 2 - qc[2] ** 2 + qc[3] ** 2),
        ]
    )


def _gravity_orientation(quat_wxyz: np.ndarray) -> np.ndarray:
    """World -Z gravity vector rotated into the frame given by quat_wxyz."""
    return quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0]))


class WbcBridge:
    """Runs decoupled_wbc's Balance/Walk ONNX policies given planar velocity commands.

    All joint-indexed arguments/returns are plain numpy arrays in the module
    order above (``ALL_JOINTS`` for state in, ``LEG_WAIST_JOINTS`` for the
    target positions out) - the caller maps those names to whatever index
    order Isaac Sim's articulation actually uses.
    """

    def __init__(self, balance_onnx: str | Path, walk_onnx: str | Path):
        global ort
        import onnxruntime as _ort
        ort = _ort
        self._balance = self._load(str(balance_onnx))
        self._walk = self._load(str(walk_onnx))
        self.reset()

    @staticmethod
    def _load(path: str):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(path, providers=providers)
        in_name = sess.get_inputs()[0].name

        def run(obs_row: np.ndarray) -> np.ndarray:
            return sess.run(None, {in_name: obs_row[None, :].astype(np.float32)})[0].squeeze(0)

        return run

    def reset(self) -> None:
        self._action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self._history = collections.deque(
            [np.zeros(SINGLE_OBS_DIM, dtype=np.float32)] * OBS_HISTORY_LEN,
            maxlen=OBS_HISTORY_LEN,
        )

    def _build_single_obs(
        self,
        qpos_all: np.ndarray,
        qvel_all: np.ndarray,
        base_quat_wxyz: np.ndarray,
        base_ang_vel_body: np.ndarray,
        loco_cmd: np.ndarray,
        height_cmd: float,
    ) -> np.ndarray:
        command = np.zeros(7, dtype=np.float32)
        command[:3] = loco_cmd * CMD_SCALE
        command[3] = height_cmd
        # command[4:7] (rpy_cmd) left at zero - no roll/pitch/yaw target exposed.

        qj_scaled = (qpos_all.astype(np.float32) - _PADDED_DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj_scaled = qvel_all.astype(np.float32) * DOF_VEL_SCALE
        gravity_orientation = _gravity_orientation(base_quat_wxyz)
        omega_scaled = base_ang_vel_body.astype(np.float32) * ANG_VEL_SCALE

        obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        obs[0:7] = command
        obs[7:10] = omega_scaled
        obs[10:13] = gravity_orientation
        obs[13 : 13 + NUM_JOINTS] = qj_scaled
        obs[13 + NUM_JOINTS : 13 + 2 * NUM_JOINTS] = dqj_scaled
        obs[13 + 2 * NUM_JOINTS : 13 + 2 * NUM_JOINTS + NUM_ACTIONS] = self._action
        return obs

    def step(
        self,
        qpos_all: np.ndarray,
        qvel_all: np.ndarray,
        base_quat_wxyz: np.ndarray,
        base_ang_vel_body: np.ndarray,
        cmd_vx: float,
        cmd_vy: float,
        cmd_wz: float,
        height_cmd: float = DEFAULT_HEIGHT_CMD,
    ) -> np.ndarray:
        """Advance the policy one control step and return new leg+waist targets.

        Args:
            qpos_all: (29,) joint positions in ``ALL_JOINTS`` order.
            qvel_all: (29,) joint velocities in ``ALL_JOINTS`` order.
            base_quat_wxyz: (4,) base orientation, world -> body, wxyz.
            base_ang_vel_body: (3,) base angular velocity expressed in the
                base's own frame (i.e. what an onboard IMU would report -
                rotate a world-frame reading through ``base_quat_wxyz`` first
                if that is what the sim provides).
            cmd_vx, cmd_vy, cmd_wz: planar velocity command (m/s, m/s, rad/s),
                straight off ``/g1/cmd_vel``.
            height_cmd: target pelvis height; defaults to the policy's
                training default.

        Returns:
            (15,) target joint positions for ``LEG_WAIST_JOINTS``, in that
            order, ready for ``set_joint_position_targets(...,
            joint_names=LEG_WAIST_JOINTS)``.
        """
        loco_cmd = np.array([cmd_vx, cmd_vy, cmd_wz], dtype=np.float32)
        single_obs = self._build_single_obs(
            qpos_all, qvel_all, base_quat_wxyz, base_ang_vel_body, loco_cmd, height_cmd
        )
        self._history.append(single_obs)
        obs = np.concatenate(list(self._history))

        if np.linalg.norm(loco_cmd) <= WALK_CMD_DEADBAND:
            self._action = self._balance(obs)
        else:
            self._action = self._walk(obs)

        return self._action * ACTION_SCALE + DEFAULT_ANGLES
