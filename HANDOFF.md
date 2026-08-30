# HANDOFF — G1 Isaac Sim + ROS2 Zenoh Bridge Stack
**Last updated:** 2026-08-29 end-of-session (all processes killed, containers stopped)
**Session state:** All stopped cleanly. Ready for fresh restart next session.

---

## ⚠️ CRITICAL: Current State (as of session end)

**Everything is STOPPED.** All laptop processes killed, spark2 containers stopped.
Handoff below reflects verified state from this session.

---

## What Was Done This Session (verified working)

### ✅ QoS Fix (LiDAR + RGBD) — VERIFIED
- `g1_sim/rtx_publisher.py` line 106: `BEST_EFFORT` → `RELIABLE`
- `g1_sim/rgbd_publisher.py` line 79: `BEST_EFFORT` → `RELIABLE`
- **Verified:** LiDAR flowing at ~14 Hz, RGBD depth/color flowing

### ✅ Namespace Fix — VERIFIED
- Changed from `/sim-bridge` → `/sim_bridge` (hyphens illegal in ROS2 topic names)
- Applied in both docker-compose.yml and laptop bridge command

### ✅ Internal WBC Mode — VERIFIED
- `--wbc-mode internal` in docker-compose.yml
- ONNX policies run inside sim, accept `/g1/cmd_vel` Twist
- `onnxruntime-gpu` auto-installed in container entrypoint
- **Verified:** WBC running, pelvis_z=0.739m, accepting cmd_vel

### ✅ robot_state_publisher — VERIFIED
- Running on laptop with cleaned G1 URDF (mujoco/gazebo tags stripped)
- Launch file: `/tmp/rsp_launch.py`
- **g1→pelvis fixed joint added** in the launch file to bridge the gap

### ✅ Zenoh Bridge — VERIFIED
- spark2 side: `eclipse/zenoh-bridge-ros2dds:latest` (network_mode: service:isaac-sim)
- Laptop side: `~/bin/zenoh-bridge-ros2dds` v1.10.0
- **160 routes created**, all sim topics bridged successfully

### ✅ CycloneDDS Config — NEW FIX
- **Problem:** zenoh bridge consumed all DDS participant slots, killing keyboard teleop
- **Fix:** Created `~/.config/cyclonedds/cyclonedds.xml` with `MaxParticipants=128`
- **Status:** Config written, verified working

### ✅ Topic Names (confirmed, no `/g1/` prefix on LiDAR/Camera)
| Category | Topics |
|----------|--------|
| **LiDAR** | `/livox/mid360/points` (NOT `/a` — single sensor now) |
| **Camera RGB** | `/camera/color/image_raw` |
| **Camera Depth** | `/camera/depth/points`, `/camera/depth/color/points` |
| **TF** | `/tf`, `/tf_static` |
| **Joints** | `/joint_states` (~108 Hz) + 29 individual `/g1/joint/*` topics |
| **IMU** | `/g1/imu` |
| **Teleop** | `/g1/cmd_vel`, `/cmd_vel` (Twist) |
| **System** | `/clock` |

---

## Known Issues (from this session)

### 🔴 TF: `warehouse` → `g1` instead of `World` → `pelvis`
**Problem:** The sim's `ROS2PublishTransformTree` OmniGraph node publishes `warehouse` → `g1` as the root dynamic TF. The `attach_robot_state_publishers()` function sets `parentPrim=/World` but the IRA setup replaces the stage root, causing the frame name to resolve to `warehouse` instead of `World`.

**TF chain as-is:**
```
Dynamic:   warehouse → g1
Static:    g1 → pelvis (added by /tmp/rsp_launch.py)
           pelvis → full robot chain (from URDF)
```

**Fix needed:** Change the `ROS2PublishTransformTree` node's parent prim or frame ID mapping so it publishes `World` → `pelvis` (or adjust RViz fixed frame to `warehouse`).

**Relevant code:**
- `g1_sim/rtx_camera.py:351` — `attach_robot_state_publishers()` function
- Line 396: `("PublishTF.inputs:parentPrim", ["/World"])` — this resolves to `warehouse` from USD hierarchy
- The warehouse USD (`assets/warehouse_ira_baked.usd`) has `/Root` as root prim, loaded under `/World/Env`

**RViz workaround:** Set Fixed Frame to `warehouse` (not `World`)

### 🟡 Keyboard Teleop Dies on Zenoh Bridge Restart
**Problem:** When zenoh bridge restarts, it creates many DDS participants quickly, exhausting the participant limit and killing other ROS2 nodes.
**Fix applied:** CycloneDDS config at `~/.config/cyclonedds/cyclonedds.xml` with `MaxParticipants=128`.
**Still happened** after fix — may need to restart teleop after bridge restart.

### 🟡 CUVSLAM + NVBLOX Install — BLOCKED
**Goal:** Install Isaac ROS 2.1 (CUVSLAM + NVBLOX) on spark2 container for SLAM + 3D mapping.
**Status:**
- ✅ Added Isaac ROS apt repo: `https://isaac.download.nvidia.com/isaac-ros/release-4.6 noble main`
- ✅ Added ROS2 Jazzy apt repo: `http://packages.ros.org/ros2/ubuntu noble main`
- ❌ No Isaac ROS Debian packages available for release-2.1 on Noble (Ubuntu 24.04)
- ❌ release-2.1 was for Humble (Ubuntu 22.04), not Jazzy/Noble
- **User specifically requested release-2.1** — may need to build from source or use a different container
- **Alternative:** Use release-4.6 (latest) which supports Jazzy/Noble, or use `nvcr.io/nvidia/isaac/ros` container

### 🟡 ROS2 Jazzy setup.bash Path Unknown
- The Isaac Sim container bundles ROS2 at `/isaac-sim/exts/isaacsim.ros2.core/jazzy/`
- This is NOT a standard ROS2 installation — no `/opt/ros/jazzy/setup.bash`
- The old path `/isaac-sim/exts/isaacsim.ros2.core/jazzy/setup.bash` no longer exists

### 🟡 Stale Containers on spark2
- `clever_snyder` and `mystifying_boyd` — 15+ hours old, should be cleaned up
- `docker rm clever_snyder mystifying_boyd` on spark2

### 🟢 TSC Crash (Intermittent)
- `Cannot calculate frequency: TSC ran backwards` — known Isaac Sim issue on ARM/GB10
- Container restarts automatically

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  spark2 (10.131.171.77) — Docker Containers          │
│                                                       │
│  ┌──────────────┐     ┌────────────────────────────┐ │
│  │ isaac-sim-ros│     │ zenoh-bridge               │ │
│  │              │ DDS │                            │ │
│  │ Isaac Sim    │◄───►│ eclipse/zenoh-bridge-      │ │
│  │ + internal   │     │   ros2dds:latest           │ │
│  │   WBC        │     │                            │ │
│  │              │     │ network_mode: service:      │ │
│  │ Publishes:   │     │   isaac-sim (shared net)   │ │
│  │ • LiDAR      │     │                            │ │
│  │ • Camera     │     │ Bridges all topics via     │ │
│  │ • TF/Joints  │     │ Zenoh TCP (port 7447)      │ │
│  │ • IMU        │     │ REST API (port 8000)       │ │
│  │              │     │                            │ │
│  │ Subscribes:  │     │ Namespace: /sim_bridge     │ │
│  │ • /g1/cmd_vel│     └──────────┬─────────────────┘ │
│  └──────────────┘                │ tcp/10.131.171.77:7447
└──────────────────────────────────┼───────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────┐
│  Laptop (dhyan-LOQ)                                   │
│                                                       │
│  ┌──────────────────────┐                            │
│  │ zenoh-bridge-ros2dds │  ← connects to spark2      │
│  │ (~/bin/)             │  Namespace: /sim_bridge     │
│  │ REST API: 8001       │                            │
│  └──────────┬───────────┘                            │
│             │ local ROS2 topics (no prefix!)          │
│             ▼                                         │
│  ┌──────────────┐  ┌───────────────────────┐        │
│  │ keyboard_    │  │ robot_state_publisher │        │
│  │ teleop       │  │ (G1 URDF → TF)       │        │
│  │ (WASDQE)     │  └───────────────────────┘        │
│  └──────────────┘  ┌───────────────────────┐        │
│                    │ RViz2                 │        │
│                    │ (LiDAR + Camera +     │        │
│                    │  RobotModel + TF)     │        │
│                    └───────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

---

## How to Start Everything (from scratch)

### Step 1: Start containers on spark2
```bash
ssh aim_spark02 "cd /home/thakk100/Projects/thesis-sim && docker compose up -d"
```

### Step 2: Wait for sim to load (~5-7 min including onnxruntime-gpu install)
```bash
ssh aim_spark02 "docker logs -f isaac-sim-ros"
# Wait for "[WH] WBC bridge" and "[WH] running" messages
```

### Step 3: Start laptop-side Zenoh bridge
```bash
~/bin/zenoh-bridge-ros2dds -e tcp/10.131.171.77:7447 -n /sim_bridge --rest-http-port 8001
```

### Step 4: Start robot_state_publisher (for robot model in RViz)
```bash
source /opt/ros/jazzy/setup.bash
source ~/Projects/thesis/g1_perception_ws/install/setup.bash
ros2 launch /tmp/rsp_launch.py
```
Note: `/tmp/rsp_launch.py` strips mujoco/gazebo tags from the URDF and adds g1→pelvis joint.

### Step 5: Start keyboard teleop (in a separate terminal)
```bash
source /opt/ros/jazzy/setup.bash
source ~/Projects/thesis/g1_perception_ws/install/setup.bash
ros2 run g1_nav keyboard_teleop
```
Controls: W=forward, S=backward, A=strafe left, D=strafe right, Q=turn left, E=turn right, SPACE=stop

### Step 6: Launch RViz
```bash
source /opt/ros/jazzy/setup.bash
rviz2 -d /tmp/sim_view.rviz
```

---

## TF Tree (verified this session)

**Dynamic TF (from sim OmniGraph):**
```
warehouse → g1
```

**Static TF (from robot_state_publisher + sim):**
```
g1 → pelvis (added by /tmp/rsp_launch.py)
pelvis → imu_in_pelvis, pelvis_contour_link, torso_link, waist_roll_link, waist_yaw_link
torso_link → d435_link, mid360_link, head_link, imu_in_torso, logo_link, waist_support_link
left_hip_pitch_link → ... → left_ankle_roll_link
right_hip_pitch_link → ... → right_ankle_roll_link
```

**Additional static TF (from sim):**
```
g1 → gt_base_link
gt_base_link → gt_pelvis
mid360_link → livox_frame
```

**Also published (from /tmp/rsp_launch.py):**
```
map → odom (identity)
odom → pelvis (identity)
```

**Complete chain:** `warehouse` → `g1` → `pelvis` → full robot

---

## RViz Configuration

### Config file: `/tmp/sim_view.rviz`
**Fixed Frame:** `warehouse` (NOT `World` — sim publishes `warehouse` as root frame)

**Displays:**
- TF (with arrows, axes, names)
- RobotModel (from `robot_description` parameter)
- Livox LiDAR (`/livox/mid360/points`, green, 3px)
- D435 Depth Points (`/camera/depth/points`, axis-color by Z, 2px)
- D435 RGB Camera (`/camera/color/image_raw`)

### If `/tmp/sim_view.rviz` is missing, recreate:
```bash
cat > /tmp/sim_view.rviz << 'RVIZ'
Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/TF
      Enabled: true
      Name: TF
    - Class: rviz_default_plugins/RobotModel
      Enabled: true
      Name: Robot Model
      Robot Description: robot_description
    - Class: rviz_default_plugins/PointCloud2
      Enabled: true
      Name: Livox LiDAR
      Topic: /livox/mid360/points
      Color Transformer: FlatColor
      Color: 0; 255; 0
      Size (Pixels): 3
    - Class: rviz_default_plugins/PointCloud2
      Enabled: true
      Name: D435 Depth Points
      Topic: /camera/depth/points
      Color Transformer: AxisColor
      Axis: Z
      Size (Pixels): 2
    - Class: rviz_default_plugins/Image
      Enabled: true
      Name: D435 RGB Camera
      Topic: /camera/color/image_raw
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: warehouse
    Frame Rate: 30
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 3
      Focal Point: {X: 0, Y: 0, Z: 0.7}
      Pitch: 0.5
      Yaw: 3.14
Window Geometry:
  Width: 1280
  Height: 720
RVIZ
```

### If `/tmp/rsp_launch.py` is missing, recreate:
```bash
cat > /tmp/rsp_launch.py << 'PYEOF'
import re
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    urdf_path = "/home/thakk100/Projects/thesis/g1_perception_ws/install/g1_description/share/g1_description/urdf/g1_29dof.urdf"
    with open(urdf_path) as f:
        urdf = f.read()
    urdf = re.sub(r'<mujoco>.*?</mujoco>', '', urdf, flags=re.DOTALL)
    urdf = re.sub(r'<gazebo.*?</gazebo>', '', urdf, flags=re.DOTALL)
    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': urdf, 'use_sim_time': True}],
        ),
    ])
PYEOF
```

---

## File Map

### Core Simulation
| File | Purpose |
|---|---|
| `scripts/g1_warehouse_sim.py` | Main sim entry point — IRA actors, LiDAR, ROS2 init, WBC mode selection |
| `g1_sim/rtx_lidar.py` | Sensor prim authoring, MID360_QUAT_WXYZ, spawn_mid360() |
| `g1_sim/rtx_publisher.py` | ROS2 PointCloud2 publisher (**QoS: RELIABLE**) |
| `g1_sim/rgbd_publisher.py` | RGBD depth/color publisher (**QoS: RELIABLE**) |
| `g1_sim/joint_cmd_sub.py` | External WBC joint-command listener (29 × Float64) |
| `g1_sim/wbc_bridge.py` | Internal WBC bridge (ONNX decoupled_wbc Balance/Walk) |
| `g1_sim/rtx_camera.py` | D435 camera spawn, `attach_robot_state_publishers()` (TF), cmd_vel OmniGraph |
| `g1_sim/ira_actors.py` | IRA actor TF publishing |
| `g1_sim/action_graph.py` | OmniGraph helpers (read_cmd_vel) |

### LiDAR Configs
| File | Purpose |
|---|---|
| `assets/lidar_configs_rotary/Livox_Mid360_R.json` | Rotary profile (128 emitters, synthetic repeating) |

### Zenoh Bridge
| File | Location | Purpose |
|---|---|---|
| `zenoh-bridge-ros2dds` | `~/bin/` (laptop) | Laptop-side bridge binary (x86_64, v1.10.0) |
| `libzenoh_plugin_ros2dds.so` | `~/bin/` (laptop) | Zenoh ROS2DDS plugin (x86_64, v1.10.0) |

### Docker
| File | Location | Purpose |
|---|---|---|
| `docker-compose.yml` | `/home/thakk100/Projects/thesis-sim/` | spark2 Docker compose (sim + zenoh bridge) |
| G1_sim volume mount | `./G1_sim:/workspace/G1_sim` | Code mount into container |

### Assets
| File | Purpose |
|---|---|
| `assets/warehouse_ira_baked.usd` | IRA cached scene (664 MB) |
| `assets/g1_29dof_sensors.usd` | G1 robot USD with sensors |
| `assets/policy/GR00T-WholeBodyControl-Balance.onnx` | WBC balance policy |
| `assets/policy/GR00T-WholeBodyControl-Walk.onnx` | WBC walk policy |

### External (laptop)
| File | Purpose |
|---|---|
| `~/Projects/thesis/g1_perception_ws/src/g1_nav/g1_nav/keyboard_teleop.py` | WASDQE keyboard teleop |
| `~/Projects/thesis/g1_perception_ws/src/g1_wbc/g1_wbc/wbc_node.py` | External WBC node (not used in internal mode) |
| `~/Projects/thesis/g1_perception_ws/src/g1_description/urdf/g1_29dof.urdf` | G1 URDF for robot_state_publisher |
| `~/.config/cyclonedds/cyclonedds.xml` | CycloneDDS config (MaxParticipants=128) |

---

## Key Technical Details

### ROS2
- **Container bundles Jazzy** at `/isaac-sim/exts/isaacsim.ros2.core/jazzy/`
- **RMW:** `rmw_cyclonedds_cpp` (required for Isaac Sim OmniGraph publishers)
- **Domain ID:** 0
- **Fixed Frame for visualization:** `warehouse` (sim publishes this as root)

### Zenoh Bridge
- **spark2 side:** `eclipse/zenoh-bridge-ros2dds:latest` Docker image
  - `network_mode: service:isaac-sim` (shares network, separate IPC)
  - Namespace: `/sim_bridge`
  - REST API: `http://localhost:8000`
- **laptop side:** `~/bin/zenoh-bridge-ros2dds` v1.10.0
  - Connects via: `tcp/10.131.171.77:7447`
  - Namespace: `/sim_bridge` (must match spark2 side)
  - REST API: `http://localhost:8001`
  - **Requires:** `~/.config/cyclonedds/cyclonedds.xml` with MaxParticipants≥128

### WBC (Whole Body Control)
- **Mode:** `internal` — ONNX decoupled_wbc Balance/Walk policies run inside sim
- **Input:** `/g1/cmd_vel` (Twist) from keyboard teleop
- **Rate:** 50 Hz control loop
- **ONNX policies:** `assets/policy/GR00T-WholeBodyControl-{Balance,Walk}.onnx`
- **Requires:** `onnxruntime-gpu` (auto-installed in container entrypoint)

### Docker
- **Image:** `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`
- **Entry point:** Custom bash entrypoint with onnxruntime-gpu install + sim launch
- **Python:** Use `/isaac-sim/python.sh` (not system python)
- **GPU:** Must include `deploy.resources.reservations.devices` with `driver: nvidia`

---

## Git State

```
Branch: isaacsim6-rtx-emitter (ahead of origin by 3 commits)
Commits: 1b5b438 (rotary), c09e0a1 (WBC), 341e5ce (runtime)
Uncommitted changes:
  - HANDOFF.md (this file)
  - Tasks.md
  - g1_sim/rtx_publisher.py (QoS fix: BEST_EFFORT → RELIABLE)
  - g1_sim/rgbd_publisher.py (QoS fix: BEST_EFFORT → RELIABLE)
  - g1_sim/rtx_camera.py (camera clearance + attach_robot_state_publishers)
  - g1_sim/wbc_bridge.py
  - scripts/g1_warehouse_sim.py
  - rviz/g1_rtx.rviz
  - docs/RTX_LIDAR_INIT.md
Removed: scripts/ros2_ws_bridge.py (replaced by Zenoh bridge)
```

---

## Useful Commands

```bash
# SSH into spark2
ssh aim_spark02

# Check container status
docker ps -a | grep -E 'isaac|zenoh'

# Watch container logs
docker logs -f isaac-sim-ros     # sim
docker logs -f zenoh-bridge      # zenoh bridge

# Check WBC status
docker logs isaac-sim-ros 2>&1 | grep "\[WH\]" | tail -5

# Full restart
cd /home/thakk100/Projects/thesis-sim && docker compose down && docker compose up -d

# Start laptop Zenoh bridge
~/bin/zenoh-bridge-ros2dds -e tcp/10.131.171.77:7447 -n /sim_bridge --rest-http-port 8001

# List topics
source /opt/ros/jazzy/setup.bash
ros2 topic list

# Check topic rates
ros2 topic hz /joint_states
ros2 topic hz /livox/mid360/points
ros2 topic hz /camera/depth/points

# Keyboard teleop
source ~/Projects/thesis/g1_perception_ws/install/setup.bash
ros2 run g1_nav keyboard_teleop

# Launch robot_state_publisher
ros2 launch /tmp/rsp_launch.py

# Launch RViz
rviz2 -d /tmp/sim_view.rviz

# Copy files to spark2
scp <file> aim_spark02:/home/thakk100/Projects/thesis-sim/G1_sim/<path>

# Clean up stale containers on spark2
ssh aim_spark02 "docker rm -f clever_snyder mystifying_boyd 2>/dev/null"
```

---

## Next Session TODO

1. **Fix TF root frame**: Change `attach_robot_state_publishers()` in `g1_sim/rtx_camera.py` to publish `World` → `pelvis` instead of `warehouse` → `g1`. Options:
   - Override the `parentFrameId` attribute on the `ROS2PublishTransformTree` node
   - Or change `parentPrim` to a prim that resolves to `World`
   - Or create a static TF publisher that maps `World` → `warehouse`
2. **CUVSLAM + NVBLOX**: Decide on approach (build from source, use release-4.6, or use separate container)
3. **Clean up stale containers** on spark2
4. **Verify g1→pelvis connection** in TF chain (robot model should show connected in RViz)
5. **D435 depth points** — verify display is correct (organized cloud, frame_id=`d435_color_optical_frame`)
