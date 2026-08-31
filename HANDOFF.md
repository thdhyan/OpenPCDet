# HANDOFF — G1 Isaac Sim + ROS2 Zenoh Bridge Stack
**Last updated:** 2026-08-31 (CUVSLAM + NVBLOX integration verified)
**Branch:** `isaacsim6-rtx-emitter` (pushed to GitHub)

---

## 🆕 CUVSLAM + NVBLOX Perception Stack — Verified 2026-08-31

### What was done
- Created `launch/perception_launch.py` — single launch file wiring CUVSLAM + NVBLOX to sim sensors
- Created `launch/depth_converter.py` — converts 32FC1 depth (metres) → uint16 (mm) for CUVSLAM RGBD mode
- Created `scripts/launch_perception.sh` — convenience script to launch from inside container
- Fixed Dockerfile: installs real `libnpp-13-0` from NVIDIA CUDA repo (Isaac ROS NVBLOX needs CUDA 13 NPP)
- Fixed entrypoint: sets `AMENT_PREFIX_PATH`, CUDA 13 library paths
- Fixed CycloneDDS config: interface `lo` (not `loopback`), removed deprecated `MaxParticipants`/`Transport` elements

### Architecture
```
Isaac Sim container (spark2)
├── g1_warehouse_sim.py (WBC + sensors)
├── depth_format_converter.py (32FC1 m → uint16 mm)
├── perception_launch.py
│   ├── static_transform_publisher (World → map)
│   ├── visual_slam_node (CUVSLAM, RGBD mode, tracking_mode=2)
│   └── nvblox_node (3D reconstruction, static TSDF)
└── zenoh-bridge-ros2dds (sidecar, forwards all topics)

Laptop
├── zenoh-bridge-ros2dds (receives all topics)
├── robot_state_publisher (URDF → TF)
└── RViz2 (visualization)
```

### Sim sensor topics consumed
| Topic | Type | Encoding | Notes |
|-------|------|----------|-------|
| `/g1/camera/rgb` | Image | rgb8 | 480×640, d435_color_optical_frame |
| `/g1/camera/depth` | Image | 32FC1 | 480×640, metres, RELIABLE QoS |
| `/g1/camera/camera_info` | CameraInfo | plumb_bob | fx=fy=460.55, cx=320, cy=240 |
| `/g1/imu` | Imu | — | pelvis frame |
| `/livox/mid360/points/a` | PointCloud2 | — | ~30k pts, RELIABLE QoS |
| `/tf` | TFMessage | — | World → pelvis → robot chain |
| `/g1/joint_states` | JointState | — | All joints |

### CUVSLAM configuration (RGBD mode)
- **tracking_mode**: 2 (RGBD) — uses `image_0` (RGB) + `depth_0` (uint16 mm)
- **depth_scale_factor**: 1000.0 (divides uint16 mm by 1000 to get metres)
- **base_frame**: pelvis
- **map_frame**: map, **odom_frame**: odom
- **camera_optical_frames**: `["d435_color_optical_frame"]`
- **IMU**: disabled (D435 has no IMU; noise params are nominal)

### NVBLOX configuration
- **voxel_size**: 0.05 m
- **mapping_type**: static_tsdf
- **global_frame**: odom
- **input_qos**: DEFAULT (works with Isaac Sim's RELIABLE publisher)
- **map_clearing_frame_id**: pelvis
- **workspace**: ±15 m, height -0.5 to 3.0 m

### CUDA 13 compatibility (CRITICAL)
Isaac Sim 4.5 ships CUDA 12.8, but Isaac ROS release-4.6 needs CUDA 13 libraries:
- **libnppidei.so.13, libnppim.so.13, libnppitc.so.13, libnppc.so.13**: Installed via `libnpp-13-0` from NVIDIA CUDA apt repo
- **libcudart.so.13**: Installed as dependency of `libnpp-13-0`
- **libnvJitLink.so.13**: From Isaac Sim's cu13 Python package at `/isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/cu13/lib/`
- **ldconfig**: `/etc/ld.so.conf.d/cuda13-cu13.conf` points to cu13 Python package lib dir

### Known issues
1. **`ros2 topic echo` doesn't work**: Python type support mismatch between apt-installed ROS2 (Python 3.12) and Isaac Sim's bundled ROS2 (Python 3.11). C++ nodes work fine; CLI tools have import errors.
2. **Frame rate warnings**: CUVSLAM expects 30 FPS but sim publishes ~20 FPS. Warnings are cosmetic.
3. **NVBLOX initial TF lookup**: First map clearing fails because CUVSLAM hasn't published TF yet. Recovers after ~1 second.
4. **Isaac Sim transient crash**: `bad_variant_access` in `omni.anim.behavior.core` — random, retry works.

### To launch the perception stack
```bash
# Inside container (after sim is running):
export PATH=/opt/ros/jazzy/bin:/usr/local/cuda-13.0/bin:$PATH
export AMENT_PREFIX_PATH=/opt/ros/jazzy
export CYCLONEDDS_URI=file:///root/.config/cyclonedds/cyclonedds.xml
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:/usr/local/cuda-13.0/lib64:/isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
ros2 launch /workspace/thesis-sim/G1_sim/launch/perception_launch.py
```

### Output topics (available via zenoh to laptop)
| Topic | Type | Source |
|-------|------|--------|
| `/visual_slam/tracking/odometry` | Odometry | CUVSLAM |
| `/visual_slam/tracking/slam_path` | Path | CUVSLAM |
| `/tf` (map→odom→pelvis) | TFMessage | CUVSLAM |
| `/nvblox_node/mesh` | MarkerArray | NVBLOX |
| `/nvblox_node/static_map_slice` | OccupancyGrid | NVBLOX |
| `/nvblox_node/esdf_slice_bounds` | MarkerArray | NVBLOX |

---

## 🆕 Isaac ROS (CUVSLAM + NVBLOX) — Installed 2026-08-31

### What was done
- Created `docker/Dockerfile` extending `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`
- Isaac ROS release-4.6 packages installed from NVIDIA apt repo (`noble-fastos` for DGX Spark ARM64)
- CUVSLAM: `ros-jazzy-isaac-ros-visual-slam` — visual SLAM node
- NVBLOX: `ros-jazzy-isaac-ros-nvblox` — 3D scene reconstruction + Nav2 costmap
- Supporting packages: nav2, tf2, pcl, rviz2, robot-state-publisher

### Key technical decisions
1. **Dummy CUDA metapackages**: Created empty `cuda-toolkit-13-0`, `tensorrt`, etc. packages via `equivs` to satisfy Isaac ROS apt dependencies without installing conflicting CUDA libraries from the NVIDIA CUDA repo
2. **Do NOT install `ros-jazzy-ros-base`**: Installing it from apt causes a TSC crash on DGX Spark (ARM64). The base image already has Isaac Sim's bundled ROS 2 Jazzy
3. **ROS 2 apt repo**: Added for PCL packages (`ros-jazzy-pcl-conversions`, `ros-jazzy-pcl-ros`)
4. **Isaac Sim's bundled ROS**: Entrypoint uses `/isaac-sim/exts/isaacsim.ros2.core/jazzy/` not `/opt/ros/jazzy/`

### Isaac ROS packages available
| Package | Purpose |
|---------|---------|
| `isaac_ros_visual_slam` | CUVSLAM visual SLAM node |
| `isaac_ros_visual_slam_interfaces` | CUVSLAM message types |
| `isaac_ros_nvblox` | NVBLOX 3D reconstruction |
| `nvblox_ros`, `nvblox_msgs` | NVBLOX ROS integration |
| `nvblox_nav2` | NVBLOX Nav2 costmap plugin |
| `ros-jazzy-nav2-costmap-2d` | Nav2 costmap |
| `ros-jazzy-robot-state-publisher` | URDF→TF publisher |
| `ros-jazzy-tf2-ros` | TF2 transform library |
| `ros-jazzy-pcl-conversions` | PCL↔ROS conversions |

### Usage
```bash
# Build the custom image
cd /home/thakk100/Projects/thesis-sim
docker compose build

# Run
docker compose up -d

# Inside container, Isaac ROS packages are available:
ros2 pkg list | grep isaac_ros
```

---

## ⚠️ CRITICAL LESSON: Zenoh Bridge Must Be Restarted With Sim

**When the sim container restarts, the laptop zenoh bridge MUST also be restarted.**
The sim-side zenoh bridge (in the container) gets a new participant ID on restart. The
old laptop-side bridge holds stale connection state and loses routes. This was the root
cause of "zero messages" on point cloud topics — NOT a QoS or CycloneDDS issue.

```bash
# On spark2:
cd /home/thakk100/Projects/thesis-sim && docker compose down && docker compose up -d

# On laptop (kill and restart):
pkill -f zenoh-bridge-ros2dds
~/bin/zenoh-bridge-ros2dds -e tcp/<SPARK_IP>:7447 -n /sim_bridge --rest-http-port 8001 &
```

---

## ✅ Verified Working (as of 2026-08-30)

| Feature | Status | Details |
|---------|--------|---------|
| **LiDAR** | ✅ ~0.5 Hz | `/livox/mid360/points/a` (rclpy, RELIABLE QoS) |
| **Depth + RGB** | ✅ ~0.5 Hz | `/g1/camera/depth/color/points` (XYZRGB, PCL convention) |
| **Depth (XYZ)** | ✅ ~0.7 Hz | `/g1/camera/depth/points` (OmniGraph depth_pcl) |
| **Joint states** | ✅ ~3 Hz | `/g1/joint_states` |
| **TF** | ✅ ~5 Hz | `World` → `pelvis` → full robot chain |
| **WBC** | ✅ | Internal mode, accepts `/g1/cmd_vel` Twist |
| **Keyboard teleop** | ✅ | strafe + walking working |
| **CycloneDDS** | ✅ | `MaxParticipants=128` on laptop |

---

## Topic Names (confirmed)

| Category | Topic |
|----------|-------|
| **LiDAR** | `/livox/mid360/points/a` |
| **Camera RGB** | `/g1/camera/rgb` |
| **Camera Depth** | `/g1/camera/depth` |
| **Depth XYZ** | `/g1/camera/depth/points` |
| **Depth XYZRGB** | `/g1/camera/depth/color/points` |
| **Joint states** | `/g1/joint_states`, `/joint_states` |
| **IMU** | `/g1/imu` |
| **cmd_vel** | `/g1/cmd_vel` |
| **TF** | `/tf`, `/tf_static` |
| **Clock** | `/clock` |
| **Robot description** | `/robot_description` |

---

## TF Tree (verified)

```
World (dynamic, from sim OmniGraph)
  └── pelvis (dynamic, from sim)
      ├── torso_link
      │   ├── head_link
      │   ├── d435_link → d435_color_optical_frame
      │   ├── left_shoulder_pitch_link → ... → left_rubber_hand
      │   ├── right_shoulder_pitch_link → ... → right_rubber_hand
      │   └── imu_in_torso
      ├── left_hip_pitch_link → ... → left_ankle_roll_link
      ├── right_hip_pitch_link → ... → right_ankle_roll_link
      ├── pelvis_contour_link
      ├── mid360_link
      └── imu_in_pelvis
```

**Key:** No artificial `g1` frame. Sim publishes `World` → `pelvis` directly.
RSP publishes `pelvis` → full chain. Fixed Frame in RViz = `World`.

---

## Docker Setup

### Files
- `docker-compose.yml` → copied to spark2 at `/home/thakk100/Projects/thesis-sim/docker-compose.yml`
- `docker/entrypoint.sh` → at `G1_sim/docker/entrypoint.sh` on spark2
- `docker/cyclonedds.xml` → reference only; NOT used via CYCLONEDDS_URI (causes rclpy crash)

### Portability
Works on DL, Spark, any NVIDIA GPU server. Just:
1. Clone the repo
2. `cd G1_sim/docker && docker compose up -d`
3. Start laptop zenoh bridge

### Entry point
The entrypoint installs `onnxruntime-gpu` and launches the sim. Isaac ROS (CUVSLAM/NVBLOX)
is NOT installed in the container — needs separate setup (see CUVSLAM section below).

---

## CUVSLAM + NVBLOX Status

**NOT yet installed.** Attempted install in container entrypoint failed:
- No `sudo` in the container (ubuntu user)
- Isaac ROS Debian packages not available for Jazzy/Noble (release-2.1 was Humble only)
- Building from source requires `colcon` and ROS2 build tools not in the container

**Options:**
1. Build a custom Docker image with Isaac ROS pre-installed
2. Use release-4.6 packages when available for Jazzy
3. Run CUVSLAM/NVBLOX as separate containers

---

## Startup Procedure

### 1. On spark2 (sim + zenoh bridge)
```bash
ssh aim_spark02
cd /home/thakk100/Projects/thesis-sim
docker compose down 2>/dev/null
docker compose up -d
# Wait ~7-10 min for sim to load
docker logs -f isaac-sim-ros | grep '\[WH\]'
```

### 2. On laptop (zenoh bridge + robot_state_publisher + teleop)
```bash
# Zenoh bridge
~/bin/zenoh-bridge-ros2dds -e tcp/10.131.171.77:7447 -n /sim_bridge --rest-http-port 8001 &

# Robot state publisher
source /opt/ros/jazzy/setup.bash
ros2 launch /tmp/rsp_launch.py &

# Keyboard teleop (interactive terminal!)
source /opt/ros/jazzy/setup.bash
source ~/Projects/thesis/g1_perception_ws/install/setup.bash
ros2 run g1_nav keyboard_teleop
```

### 3. RViz (optional)
```bash
source /opt/ros/jazzy/setup.bash
rviz2 -d /tmp/sim_view.rviz
# Fixed Frame: World
# Add: LiDAR (PointCloud2), Depth RGB (PointCloud2), RobotModel, TF
```

---

## Code Changes Made This Session

| File | Change | Line |
|------|--------|------|
| `g1_sim/rtx_publisher.py` | QoS: BEST_EFFORT → RELIABLE | 106 |
| `g1_sim/rgbd_publisher.py` | QoS: BEST_EFFORT → RELIABLE | 79 |
| `/tmp/rsp_launch.py` | Removed artificial `g1→pelvis` joint | all |
| `docker/entrypoint.sh` | Simplified: onnxruntime + sim only | all |
| `docker/docker-compose.yml` | Portable, thesis-sim volume mount | all |
| `docker/cyclonedds.xml` | Reference: MaxParticipants=128 | new |

---

## Known Issues
- **LiDAR rate is ~0.5 Hz** (not the ~14 Hz from earlier session). The sim LiDAR
  publisher runs at sim frame rate but zenoh bridge forwarding adds latency.
- **Isaac ROS CUVSLAM/NVBLOX** not installed — blocked by container permissions
- **No CYCLONEDDS_URI** in container — setting it via env var causes rclpy Node crash
- **Carters_0 agent** fails to load nova_carter.yaml (harmless warning)
