# HANDOFF — G1 Isaac Sim + ROS2 Zenoh Bridge Stack
**Last updated:** 2026-08-30 end-of-session
**Branch:** `isaacsim6-rtx-emitter` (pushed to GitHub)

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
