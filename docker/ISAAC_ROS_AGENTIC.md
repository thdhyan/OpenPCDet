# Isaac ROS + AgenticROS on `dl`

## Compatibility decision

Run this stack on **`dl` (x86_64)**, not on Spark02:

- Isaac ROS **4.5** is the current build target because it uses ROS 2 Jazzy and
  CUDA 13.0 (driver 580+), which is a valid fit for the `dl` driver line.
- Isaac ROS 4.6/5.0 packages currently resolve to CUDA 13.1+/13.2+ runtime
  dependencies. `dl` reports driver `590.48.01` and CUDA `13.1`; Isaac ROS 5.0
  requires driver 595+ and is therefore not an official match for this host.
- Isaac ROS 4.5 was not the release line officially validated with Isaac Sim
  6.0, so the Isaac Sim 6.0 ↔ 4.5 pairing remains a runtime compatibility
  gate. Do not upgrade to 4.6/5.0 merely to match Isaac Sim 6.0 without first
  updating the host driver and re-running the checks below.
- cuVSLAM is supported on x86; it is explicitly not supported on DGX Spark.

The existing `docker/Dockerfile` builds the Isaac Sim 6 image with the Isaac
ROS 4.5 `visual_slam` and `nvblox` packages and pins apt to CUDA 13.0. Do not
replace it with the Spark or Isaac ROS 5 container.

## Topic contract

`launch/perception_launch.py` is already wired to the current G1 graph:

| Consumer | Isaac Sim input |
|---|---|
| CuVSLAM RGBD mode | `/g1/camera/rgb`, `/g1/camera/camera_info`, converted `/g1/camera/depth_uint16`, `/g1/imu` |
| NVBlox | `/g1/camera/depth`, `/g1/camera/camera_info`, `/g1/camera/rgb`, `/tf` |
| Clock/TF | `/clock`, `/tf`, `/tf_static` |

RGBD mode accepts one aligned RGBD camera; a stereo pair is not required. The
depth converter converts Isaac Sim `32FC1` metres to `16UC1` millimetres.
NVBlox is intentionally camera-only for now: the Mid-360 publisher emits one
PointCloud2 per beam (`/livox/mid360/points/a` through `/d`), while NVBlox
expects one merged lidar topic. Add a merger before enabling `use_lidar`.

Outputs include:

- `/visual_slam/tracking/odometry`
- `/visual_slam/tracking/slam_path`
- `/nvblox/gpu_mesh`
- `/nvblox/esdf_distance_slice`
- `/nvblox/costmap`

## Start order

All GPU services use `ISAAC_ROS_GPU` (default `1`) and are opt-in profiles.
Do not start them while the selected GPU is occupied.

```bash
# CPU-only AgenticROS transport; safe to run while other GPU jobs are active
docker compose -f docker/docker-compose-isaac-ros.yml \
  --profile agent up -d agenticros-rosbridge

# After selecting a free GPU:
ISAAC_ROS_GPU=1 docker compose -f docker/docker-compose-isaac-ros.yml \
  --profile sim --profile perception up -d

# Run the AgenticROS MCP stdio server for Codex/Claude/Hermes
docker compose -f docker/docker-compose-isaac-ros.yml \
  --profile agent-mcp run --rm agenticros-mcp
```

The AgenticROS container is configured for the G1 graph through
`ws://127.0.0.1:9091` (port 9090 is already used by another cockpit service
on `dl`); its config is in `docker/agenticros/agenticros-config.json`. It uses
rosbridge, not Gazebo, so there is only one simulation source: Isaac Sim.

## Validation

```bash
docker compose -f docker/docker-compose-isaac-ros.yml ps
curl http://127.0.0.1:9091
# inside the Isaac ROS perception container:
source /opt/ros/jazzy/setup.bash
ros2 topic list | grep -E 'camera|visual_slam|nvblox|imu|tf'
ros2 topic echo /visual_slam/tracking/odometry --once
ros2 topic echo /nvblox/esdf_distance_slice --once
```

Do not claim CuVSLAM/NVBlox runtime success until those topics publish and the
container logs show no missing-frame, incompatible-CUDA, or NITROS errors.
