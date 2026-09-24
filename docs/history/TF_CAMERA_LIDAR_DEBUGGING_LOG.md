> **Superseded (2026-09-23):** historical notes. Current state and resume steps live in `HANDOFF.md` (Workstream A) and `docs/TACSL_PLAN.md`.

# TF / Camera / Lidar Debugging Log — G1 Isaac Sim

**Date**: 2026-09-23  
**Status**: NOT FIXED — Root causes identified, fixes partially applied, alignment still broken  
**Session State**: `SESSION_STATE_20260923.md` has broader context

---

## 1. Coordinate Frames Reference (URDF → USD)

| Frame | Parent | Origin XYZ (m) | RPY | Notes |
|-------|--------|----------------|-----|-------|
| `pelvis` | World | (0,0,0) | (0,0,0) | Root |
| `waist_yaw_link` | pelvis | (0,0,0) | (0,0,0) | Revolute Z |
| `waist_roll_link` | waist_yaw | (-0.00396, 0, 0.035) | (0,0,0) | Revolute X |
| `torso_link` | waist_roll | (0,0,0) | (0,0,0) | Revolute Y |
| `mid360_link` | torso_link | (0.0002835, 0.00003, 0.4188) | **(3.14, 0, 0)** | Fixed — 180° roll baked in |
| `d435_link` | torso_link | (0.05762, 0.01753, 0.41987) | (0, 0.8308, 0) | Fixed — ~47.6° pitch |
| `imu_in_torso` | torso_link | (-0.0396, -0.00224, 0.1379) | (0,0,0) | Fixed |
| `imu_in_pelvis` | pelvis | (0.04525, 0, -0.0834) | (0,0,0) | Fixed |

**Key insight**: The URDF's `mid360_joint` has `rpy="3.14 0 0"` (180° roll). This is baked into the USD's `mid360_link` prim transform. The sensor prims should mount with **identity orientation** — NOT another 180° roll.

---

## 2. What Was Broken (Initial State)

### Lidar
- `spawn_mid360()` defaulted to `MID360_QUAT_WXYZ = (0, 1, 0, 0)` — 180° roll
- USD already had 180° roll from URDF → **double roll** on sensor prims
- TF tree: `mid360_link` had correct orientation, but sensor points came out rotated
- Result: Points appeared below ground (z = -6m in world)

### Camera Depth (`/g1/camera/depth/points`)
- OmniGraph `depth_pcl` helper output stamped `d435_color_optical_frame`
- But annotator outputs in USD camera local frame (X-right, Y-down, Z-forward ≠ optical frame)
- No static TF from `d435_link` → camera prim frame
- Result: Depth points misaligned with colorized cloud (`/g1/camera/depth/color/points`)

---

## 3. Fixes Applied (In Order)

### Fix 1: Lidar identity orientation in `rtx_lidar.py`
```python
# OLD default
orientation: tuple = MID360_QUAT_WXYZ  # (0, 1, 0, 0) = 180° roll

# NEW default  
_IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
orientation: tuple = _IDENTITY_QUAT_WXYZ
```
**File**: `g1_sim/rtx_lidar.py` line ~306

### Fix 2: Warehouse calls identity in `g1_robot.py`
```python
# OLD
orientation=lidar_orientation if lidar_orientation is not None else MID360_QUAT_WXYZ

# NEW
orientation=lidar_orientation if lidar_orientation is not None else (1.0, 0.0, 0.0, 0.0)
```
**File**: `g1_sim/g1_robot.py` line ~349

### Fix 3: Camera depth_pcl frame_id + static TF in `rtx_camera.py`
```python
# depth_pcl helper uses camera prim frame
depthpoints_frame = "d435_camera" if label == "DepthPoints" else frame_id
values += [(f"{node}.inputs:frameId", depthpoints_frame)]

# Static TF: d435_link -> d435_camera (camera prim's local frame)
# Published by ROS2PublishRawTransformTree
cam_tf = (-0.5, -0.5, -0.5, 0.5)  # xyzw for R_z(-90)*R_y(-90)
```
**File**: `g1_sim/rtx_camera.py` lines ~223-251

---

## 4. Current Verification Results (After Fixes)

### TF Tree (from `/tf` + `/tf_static`)
```
World
└── pelvis (0.739 Z) ✓
    └── ... → torso_link
        ├── mid360_link: pos=(-0.302, 0.143, 1.211) ✓ (Z=1.211 matches URDF 0.8+0.4188)
        ├── d435_link: pos=(-0.243, 0.151, 1.212) ✓
        ├── d435_camera: NOT FOUND in /tf  ❌
        └── d435_color_optical_frame: pos=(-0.243, 0.151, 1.212) ✓
```

### Point Cloud World-Frame Ranges (after TF transform)

| Topic | Source Frame | World Z Range | Mean Z | Status |
|-------|--------------|---------------|--------|--------|
| `/livox/mid360/points/a` | mid360_link | [-0.039, 3.665] | 1.74 | **Lidar OK** — floor at ~0 |
| `/g1/camera/depth/points` | d435_camera* | [-2.49, 4.84] | 1.18 | **DEPTH BROKEN** — goes below floor |
| `/g1/camera/depth/color/points` | d435_color_optical_frame | [-0.39, 0.64] | -0.13 | **COLOR OK** — floor at ~0 |

\* `d435_camera` frame NOT found in TF — static TF not publishing

### Static TF Check (`/tf_static`)
```
d435_link -> d435_color_optical_frame: rot=(-0.5, 0.5, -0.5, 0.5) ✓ (REP-103)
d435_link -> d435_camera: NOT PRESENT ❌
```

---

## 5. Root Causes Still Unresolved

### Issue A: Camera static TF not publishing
The `ROS2PublishRawTransformTree` node for `CameraTF` (d435_link → d435_camera) is added to the graph but **not emitting** on `/tf_static`. The node exists in the OmniGraph but no static transform appears.

**Suspected causes**:
- Node execution order / tick not firing
- `staticPublisher: True` but no initial publish on graph start
- Camera prim path mismatch (`d435_camera` vs actual prim name)

### Issue B: Depth points go below floor (z = -2.5m in world)
Even if TF worked, the depth_pcl annotator outputs in USD camera local frame where:
- USD camera: -Z = forward, +X = right, +Y = up
- d435_link: +X = forward, +Y = left, +Z = up
- The rotation `R_z(-90) * R_y(-90)` = (-0.5, -0.5, -0.5, 0.5) in xyzw
- But `D435_PITCH_RAD = 0.8308` (~47.6°) tilt is **baked into camera prim xform**, not in the static TF

### Issue C: `d435_camera` frame missing from TF
The camera prim is spawned at `/World/G1/torso_link/d435_camera`. The `ROS2PublishTransformTree` should publish its world transform as `d435_camera` under `torso_link`, but it's not appearing.

---

## 4. Files Modified

| File | Changes |
|------|---------|
| `g1_sim/rtx_lidar.py` | Default orientation = identity (line 306) |
| `g1_sim/g1_robot.py` | Warehouse calls identity orientation (line 349) |
| `g1_sim/rtx_camera.py` | DepthPoints frame_id = "d435_camera"; added CameraTF static TF (lines 223-251) |

---

## 5. How to Debug Next

### Check if CameraTF node exists and executes
```bash
# In sim, check OmniGraph /ActionGraph/CameraROS2 for CameraTF node
# Verify it has exec connection from OnTick
```

### Verify camera prim name
```python
# In sim, check actual prim path:
# /World/G1/torso_link/d435_camera  (or similar)
# The ROS2PublishTransformTree targetPrims should include this
```

### Test static TF manually
```bash
ros2 topic echo /tf_static --once | grep -A 20 "d435_camera"
```

### Check depth_pcl annotator output frame
The `depth_pcl` annotator in Isaac Sim outputs in the **render product's camera local frame**. The frame_id should match whatever frame the camera prim publishes on `/tf`.

---

## 6. Quick Test Commands

```bash
# 1. Check TF tree has d435_camera
ros2 run tf2_tools view_frames  # or use the Python TF listener script

# 2. Check depth/points frame_id
ros2 topic echo /g1/camera/depth/points --once | head -5

# 3. Check static TF for camera
ros2 topic echo /tf_static --once | grep -A 10 "d435_camera"

# 4. Transform depth points to world manually (Python)
# Use the verification script pattern from above
```

---

## 7. Expected Correct State

| Frame | World Z | Notes |
|-------|---------|-------|
| `mid360_link` | ~1.21 | 0.8 + 0.4188 |
| `d435_link` | ~1.21 | 0.8 + 0.41987 |
| `d435_camera` | ~1.21 | Same as d435_link (fixed offset) |
| `d435_color_optical_frame` | ~1.21 | Same origin, rotated |
| Floor | 0.0 | Ground plane |

| Point Cloud | World Z Range | Mean Z |
|-------------|---------------|--------|
| Lidar | [0.0, ~4.0] | ~1.5 |
| Depth (camera) | [0.0, ~4.0] | ~1.5 |
| Color (optical) | [0.0, ~4.0] | ~1.5 |

All three should overlap in RViz (Fixed Frame = World).

---

## 8. Current Running Sessions

```bash
tmux ls
# g1sim    - warehouse sim (running)
# g1rviz   - RViz with all displays
# g1teleop - keyboard WBC control

# Logs
/tmp/opencode/warehouse_gui10.log  # sim
/tmp/opencode/rviz.log             # rviz
```

---

## 9. Key Files to Inspect

1. `g1_sim/rtx_lidar.py` — `spawn_mid360` default orientation
2. `g1_sim/g1_robot.py` — `load_g1` lidar spawn call
3. `g1_sim/rtx_camera.py` — `attach_camera_publishers` OmniGraph construction
4. `assets/g1_29dof_sensors.usd` — USD with baked transforms
5. `rviz/g1_rtx.rviz` — RViz config (Fixed Frame = World)

---

**Next Step**: Debug why `CameraTF` static TF isn't publishing. Check OmniGraph execution and camera prim name match.
<tool_call>
<function=shell>
<parameter=command>
cd /home/thakk100/Projects/thesis/G1_sim
cat TF_CAMERA_LIDAR_DEBUGGING_LOG.md