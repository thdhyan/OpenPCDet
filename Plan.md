# Plan — design and rationale

Why this project is shaped the way it is. Live status → [Tasks.md](Tasks.md).
Quickstart / current gotchas → [HANDOFF.md](HANDOFF.md).

---

## Goal

Unitree G1 (29 DOF) carrying a Livox Mid-360 LiDAR and a D435 camera,
standing/walking via a whole-body controller, in a populated warehouse, all
published over ROS2 for RViz visualisation and 3D pedestrian detection.

## Environment (measured on this machine, not copied from docs)

| Item | Value |
|---|---|
| Python env | uv venv `/generalSSD/IsaacLab/isaac6/.venv`, Python 3.12 (Isaac Sim 6.1.0, pip-installed) |
| Isaac Lab | 3.0 EA (`release/3.0.0`) checkout `/generalSSD/IsaacLab-release-3.0.0`, editable install in the venv |
| ROS2 | jazzy — in-sim bundled rclpy via `isaacsim.ros2.bridge`; system `/opt/ros/jazzy` only for out-of-sim tools (`~/.bashrc` does NOT source it) |
| torch | 2.12.0+cu130 (measured) |
| GPU | RTX 4060 Laptop **8 GB** (Ada, sm_89) — only one Isaac Sim instance fits |

## Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| USD asset | auto-convert from URDF (`scripts/convert_g1_urdf_to_usd.py`) | mounts (`mid360_link`, `d435_link`) already exist in the URDF |
| LiDAR | **RTX LiDAR** (`isaacsim.sensors.experimental.rtx`), emitter-state JSON configs | ray-traces the real scene; the earlier warp `LidarSensor`/IsaacLab path was deleted entirely (see below) |
| Camera | RTX D435, RGB+depth+semantic | standard Isaac Sim camera render product |
| Locomotion | NVIDIA `decoupled_wbc` (Balance/Walk ONNX policies), driven by `/g1/cmd_vel` | plain Python/ONNX+numpy, fits directly — see history below |
| Warehouse population | Isaac Replicator Agent (IRA): wandering humans + Nova Carters | gives the LiDAR/camera dynamic targets, not just static geometry |
| Transport | in-sim `rclpy` (bundled jazzy rclpy inside `isaacsim.ros2.bridge`) for LiDAR; OmniGraph ROS2 nodes for everything USD-prim-backed (`/tf`, `/clock`, joint states, camera, IMU, cmd_vel) | LiDAR data lives in a tensor, not a USD prim — no OmniGraph node can see it |
| Detection | separate process, port of `livox_detection` (CenterPoint) | inference must not stall physics; different dependency set |

**Transport note**: `isaacsim.ros2.bridge` loads its own bundled jazzy rclpy
into the sim process even though system rclpy is 3.12 — no IPC sidecar
needed, just enable the extension before `import rclpy`.

## History — how we got to the current design

**Original design (Isaac Sim 5.1, `env_isaaclab` env)**: OmniPerception's warp
`LidarSensor` via IsaacLab. Passed its verification gate but had two sharp
edges worth remembering if anyone ever reads warp lidar code again:
- `samples` selects a **time window** into the `.npy` scan pattern, not a
  uniform subsample — the first 8,000 rows of `mid360.npy` are one upward
  sweep with **zero** downward rays. Use `downsample` (strides) instead of
  shrinking `samples`.
- The Mid-360 has **no straight-down ray** (steepest tilt −7.2°) — blind
  radius ≈ 8× mount height. At the G1's 0.8 m mount, that's a 6.4 m blind
  cone; near-field sensing has to come from the camera.

**Pivot to RTX LiDAR (Isaac Sim 6.0.1, `isaac` env)**: per IsaacSim GitHub
discussion #685. The warp/IsaacLab `ray_caster` path and its scripts were
later deleted outright (`g1_sim/lidar_publisher.py`, `g1_sim/robots/`,
`scripts/g1_ros2_sim.py`, `scripts/00_verify_warp_lidar.py`) — RTX fully
replaces it, no dual-path to maintain.

**decoupled_wbc locomotion**: three false starts before landing here —
`unitree_sim_isaaclab`'s `policy.onnx` turned out to be stationary-arm
manipulation, not locomotion; IsaacLab's `Isaac-Velocity-Flat-G1-v0` has no
shipped checkpoint and targets the wrong (23-DOF) config; NVIDIA GEAR-SONIC
needs a full TensorRT C++ build and mocap-style references, more than the
task needed. `decoupled_wbc` (part of `GR00T-WholeBodyControl`) is plain
Python, ONNX+numpy, driven by planar velocity commands — fit directly.
Two real bugs found integrating it: (1) NVIDIA's own vendored MuJoCo config
pointed at stale internal checkpoint filenames, not the released
`GR00T-WholeBodyControl-{Balance,Walk}.onnx`; (2) `convert_g1_urdf_to_usd.py`
bakes a **uniform** PD drive (kp=100/kd=10) into every joint, but the policy
was trained assuming much stiffer per-joint gains (kp up to 250) — the robot
collapsed in ~2 s until `Articulation.set_gains()` was called explicitly at
startup with the policy's real training gains. **Any externally trained
policy will hit this same trap against this USD** until the converter bakes
in real per-joint gains instead of a uniform placeholder. Full trace:
`docs/decoupled_wbc_findings.md`.

**Warehouse + IRA**: two real bugs found. (1) The original Nucleus-room
loader passed a catalog-relative path straight into `add_reference_to_stage`
without resolving it — silently failed every time, always fell back to flat
ground; fixed in `g1_sim/warehouse.py`. (2) IRA's own
`ensure_navmesh_ready()` polls with a **hard-coded 100-frame cap**, but the
real warehouse mesh (291+ meshes) genuinely needs ~600–700 frames to bake —
every attempt failed with a bogus error until the poll budget was patched up
to 3000 frames in `g1_sim/ira_actors.py`.

**Below-ground LiDAR bug (fixed 2026-08-11)**: RViz showed the whole cloud
below the floor. Root cause: the RTX sensor was mounted on `torso_link` with
a hand-authored 180°-roll quaternion, but points were published as
`frame_id="mid360_link"` — a *different* prim. `GenericModelOutput`'s
`frameOfReference` is `SENSOR` (confirmed live), so points come out in the
sensor prim's own frame; RViz then applied `mid360_link`'s TF (which already
carries its own 180° roll from the URDF) a *second* time, flipping every
+elevation return to −elevation. **Fix**: mount the sensor *as* the USD's
existing `mid360_link` prim with an **identity** local transform, so the
URDF-baked pose is the only transform applied, matching the published frame
exactly (`g1_warehouse_sim.py`: `mount = f"{ROBOT_PRIM}/mid360_link"`,
`translation=(0,0,0)`-ish, `orientation=(1,0,0,0)`).

**Decoupled from OmniPerception (2026-08-11)**: the external checkout is no
longer a runtime dependency. URDF/meshes/scan-pattern vendored under
`assets/` (git-ignored — copy `assets/robot/` and `assets/scan_patterns/`
manually if another machine needs them).

**charset_normalizer boot crash (fixed 2026-09-22, Isaac Sim 6.1 upgrade)**:
six failed boots before root cause — Kit's ext manager re-imports
`charset_normalizer` mid-boot from its pip prebundles, and when the compiled
`cd.so` pair initialized against `md.py` **executed from source** (28-slot
`CharInfo=240`, plain `MessDetector=16`), cython init aborted
("MessDetectorPlugin size changed … Expected 24 … got 16") and cascaded into
`isaacsim.core.api` / `sensors.experimental.rtx` import failures. Run #6's
signature was reproduced exactly by forcing source-exec in isolation; a
`sys.addaudithook` trace then showed which path each import took. Fix: all
five on-disk charset copies byte-synced to one compiled 3.5.1 set, and
`scripts/g1_warehouse_sim.py` defensively pre-imports
`charset_normalizer{,.api,.md}` *before* `from isaacsim import SimulationApp`
(the audit-hook tripwire stays in place to catch any future source re-import).
⚠ The site-packages copy is now 3.5.1 against isaacsim_kernel's `==3.3.2`
pin — **do not `uv sync` this venv** (original 3.3.2 survives in
`~/.cache/uv/archive-v0/hks5mxSSeqdWVTDD/`). Runs #7/#8 clean: 19.5 s to app
ready (was 43–96 s), 200 steps exit 0, and every ROS 2 topic (RGB, depth,
`/livox/mid360/points/a`, `/tf`, `/g1/imu`, `/g1/joint_states`) verified live
from `/opt/ros/jazzy`.

## Open technical risk

RTX LiDAR coverage is a **partial band, not a full 360° ring** even after
fixing two upstream authoring bugs (missing `numLines`/`numRaysPerLine` on
the USD prim, and an all-zero `bank` field) that previously caused 0 points.
Live warehouse runs show azimuth clustered ~50°–180°, elevation ~0°–40°,
against real warehouse mesh geometry (not a missing-geometry artifact) —
each individual `emitterState` JSON genuinely spans the full
`-180..180`/`-5.7..50` range, so something drops roughly half the rays
between the authored USD attributes and the returned `GenericModelOutput`
hits. Leading unconfirmed hypothesis: self-occlusion from the mount's
180°-roll orientation. See `docs/RTX_LIDAR_INIT.md` and Tasks.md for the
next diagnostic step.

## Detection port (CenterPoint)

Upstream `livox_detection` requires Python 3.8/torch 1.8.2/CUDA 10.2, which
has no Ada (sm_89) support — only the checkpoints are portable. The network
was re-declared against torch 2.7 by reverse-engineering the checkpoint's
275 tensors, then checked against upstream `resfpn.py`/`boolmap.py`/
`centerhead.py`, which caught five errors inspection alone missed (wrong
stride on block 0, neck upsamples every level instead of downsampling to a
common stride, `softmax` not `Sigmoid` on `attention_w`, BatchNorm
`eps=1e-3, momentum=0.01`, decoding uses `feature_map_stride=1` with `rot`
as `(cos, sin)`). Validated: checkpoint loads with 0 missing/0 unexpected
keys across all 275 tensors; recovers all 3 planted pedestrians to within
~0.1 m at ~189 ms/frame on a synthetic scene. A `ClusteringBackend` fallback
(no torch/GPU/weights) exercises the same ROS2 graph and RViz config.

## OpenPCDet comparison (not integrated)

A sibling checkout (`~/Projects/Thesis/OpenPCDet`, capital-T) has real
pretrained weights (`pointpillar_7728.pth`, 77.28 KITTI AP) that our
from-scratch `g1_perception_ws` PointPillar doesn't have. `pcdet` smoke-tests
clean end to end in a separate `livox` conda env. Different backbone shape
means the checkpoint can't load directly into our model class — two
integration paths scoped (wrap OpenPCDet as a new backend, vs. reshape our
model to match its architecture), neither implemented. Wrapping is the
lower-effort path since the smoke test already answers "does this component
work" — the only open question is whether a KITTI-forward-range model is
useful on a 360° pedestrian scene.

## SLAM (not integrated)

`sjtuyinjie/Ultra-Fusion` (LiDAR+camera+IMU, matches our exact sensor pair)
has no public source yet but its prebuilt Docker images run correctly on
this machine despite targeting a different ROS distro (Humble/22.04 vs. our
Jazzy/24.04) — Docker's userspace isolation sidesteps that. What's missing
is a sensor profile: none of its shipped configs assume a legged, wheel-less
robot, and we don't have real Mid-360/D435 extrinsics calibrated yet.
Alternatives if this stalls: FAST-LIO2/Point-LIO (LiDAR-inertial only,
Livox-native, lower risk), FAST-LIVO2 (closest public match to our exact
sensor triplet). Full trace: `docs/ultra_fusion_findings.md`.
