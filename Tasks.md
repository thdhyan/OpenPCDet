# Tasks

Status board for the G1 + Livox Mid-360 + camera + WBC + warehouse pipeline.
Design rationale/history → [Plan.md](Plan.md). Run commands/gotchas →
[HANDOFF.md](HANDOFF.md).

Legend: ✅ done · 🔄 in progress · ⛔ blocked · ⬜ not started

---

## Core pipeline (`scripts/g1_warehouse_sim.py`)

- ✅ **G1 USD** — auto-converted from URDF (`scripts/convert_g1_urdf_to_usd.py`
  → `assets/g1_29dof_sensors.usd`), all 29 DOF + `mid360_link`/`d435_link`
  mounts preserved.
- ✅ **RTX Mid-360 LiDAR** — spawned on `mid360_link` (identity local
  transform — see Plan.md for the below-ground-bug fix), publishes
  `/livox/mid360/points` via in-sim rclpy (`g1_sim/rtx_publisher.py`,
  `RtxLidarPublisher`). NOT the OG `ROS2RtxLidarHelper` graph — that
  advertises the topic but never emits on this Isaac Sim build.
- ✅ **D435 camera** — RGB/depth/semantic + camera_info
  (`/g1/camera/{rgb,depth,semantic,camera_info}`), plus a synthesized
  `/g1/camera/depth/color/points` (`g1_sim/rgbd_publisher.py`).
- ✅ **decoupled_wbc locomotion** — Balance/Walk ONNX policies via
  `g1_sim/wbc_bridge.py`, driven by `/g1/cmd_vel` at 50 Hz. Runs by default
  (no flags needed); `--no-locomotion` to disable, `--freeze-robot` to hold
  the spawn pose instead (gravity disabled, for sensor-only testing).
- ✅ **IMU** — `g1_sim/rtx_camera.py` spawn/attach → `/g1/imu`.
- ✅ **Robot state** — `/tf`, `/g1/joint_states`, `/clock` via
  `attach_robot_state_publishers`.
- ✅ **Warehouse + IRA actors** — real warehouse USD (`g1_sim/warehouse.py`,
  Nucleus-path bug fixed) + wandering humans/Nova Carters
  (`g1_sim/ira_actors.py`, navmesh poll-budget bug fixed). `--cache-scene`
  loads a pre-baked scene (`assets/warehouse_ira_baked.usd`) to skip the
  ~60–80 s warehouse-download + navmesh-bake on every run — actors sit
  static at their saved pose in that path (not verified either way whether
  wander behavior resumes; use `--rebake` when actor motion matters).
  `--no-ira` falls back to a plain warehouse + static pedestrian boxes.
- ✅ **Runs cleanly headless** — verified end to end multiple times,
  including a 120 s run with 2 IRA humans + 1 Nova Carter + full sensor
  suite + WBC.
- ✅ **Diag/debug scripts removed** (2026-08-11) — `scripts/debug_prims.py`,
  `debug_ros2_bridge.py`, `debug_rtx_graph.py`, `diag_lidar_debug.py`,
  `diag_lidar_elevation.py`, `diag_lidar_points_stats.py`,
  `diag_lidar_worldz.py`, `diag_warehouse_lidar.py` all deleted. The
  production entrypoint is `scripts/g1_warehouse_sim.py` only
  (`g1_rtx_sim.py` remains as the bare-warehouse-free alternative scene per
  its own docstring).

## Scene presets (`--env`) + VLM caption — 2026-09-24

- ✅ **Preset framework** — `--env` picks from `g1_sim/environments.py`
  (1–2) / `g1_sim/nav_environments.py` (3–5). Props go through
  `spawn_usd_prop` (raises on an unresolved reference), `make_rigid`
  (edits existing bodies, never nests), and every prop gets a semantic
  class label for segmentation/captioning. `--log-props` prints PhysX
  poses every 300 steps. Committed `a8e7e38`; evidence in
  `docs/screenshots/envs_20260924/README.md`.
- ✅ **Env 1 `tabletop_wheel`** — wheel URL 404 (`{root}/IsaacLab/…` →
  `{root}/Isaac/IsaacLab/…`) and nested-rigid-body bugs fixed; wheel stable
  at (−0.350, 0.450, 0.698) for 1000+ steps, in the D435 view,
  `verify_sensor_tf` 11/11, `check_sensor_suite` PASS.
- ✅ **Env 2 `tabletop_cluster`** — 10 labelled props (wheel, R/G/B cubes,
  YCB mug/can/bottle/banana/brick); YCB up-axis −90° fix; stable 800+ steps,
  both sensor checks pass, caption names all 10 objects.
- 🔄 **Env 3 `nav_people`** — 6 IRA walkers at fixed spawn points, navmesh
  Exclude hole carved at the robot + rebake (nearest walkable point 0.40 m);
  **verified standing stable pelvis_z ≈ 0.72 m for 400+ updates with IRA humans;
  no fall — floor/box obstacles are the fall cause for env 4**; screenshots captured,
  sensor checks pending.
- ✅ **Env 4 `nav_people_boxes`** — + 0.8 m/10 kg and 0.3 m/1 kg boxes with
  their own navmesh holes (3 holes rebaked); verified standing stable pelvis_z ≈ 0.72–0.73 m for 150+ updates
  with new box positions (y=3.5 m, y=4.5 m); 6 perspective captures saved. Fall cause isolated to
  box placement; navmesh hole ∼1.6 m per finding #7.
- ⬜ **Env 5 `nav_people_forearm_box`** — box welded to both forearms
  (FixedJoints excluded from the articulation, elbow lower limit raised to
  the spawn angle, robot↔box collision filtered). Code written, **not yet
  run**; `g1_sim/nav_environments.py` still has TEMP debug (`_dbg_box_watch`)
  to delete after verification.
- ✅ **VLM caption** — `--caption DIR` runs `isaacsim.replicator.caption.core`
  once at `--capture-step` (`g1_sim/vlm_caption.py`): OpenAI `gpt-6-luna`
  ~2 s/req after in-process IRC patches (no NVIDIA-key leak to OpenAI, dropped
  rejected params, per-prop scene-graph nodes). NVIDIA endpoint is slow
  (kimi-k3 ~67 s, deepseek >150 s, cosmos/nemotron 404).
- ⛔ **One sim at a time** — 8 GB GPU; runs serialize through
  `flock /tmp/opencode/g1sim.lock`. Never start a second sim.

## RTX LiDAR coverage — three root causes found and fixed ✅ (2026-08-13)

**Root causes (all three present simultaneously):**

1. **`elementsCoordsType=SPHERICAL` (GMO default)** — `gmo.x/y/z` are
   azimuth(deg)/elevation(deg)/range(m), not Cartesian x/y/z. Publisher
   stacked them as `[x, y, z]` metres, so elevation values in degrees (+11°…
   +87°) were placed at those metre offsets — producing the "upper-hemisphere
   band" symptom. Fixed: `spawn_mid360` now sets
   `omni:sensor:Core:elementsCoordsType="CARTESIAN"` on every prim; publisher
   has runtime check + SPHERICAL→Cartesian fallback.

2. **Elevation-biased chunk split in `gen_mid360_rtx_config.py`** — the
   `mid360.npy` pattern is stored sorted by elevation within each frame.
   Contiguous spatial chunks (prim 0 = points 0–4999, etc.) therefore landed
   on elevation sub-bands: prims A & D got `el=[+17°…+52°]` only (upper
   hemisphere → nothing but ceiling → 0 hits), while B & C got
   `el=[-7°…+22°]`. Fixed: per-frame shuffle with `rng.shuffle` before
   chunking; all 4 prims now cover `el=[-7°…+52°]` with ~15% negative
   (ground) rays. Configs regenerated (25 states, 4.73 MB each, under 5 MB
   Hydra limit).

3. **Missing 180° roll on spawned sensor prims** — `mid360_link` in the USD
   carries the URDF's `rpy=(3.14, 0, 0)` (180° roll), making the link's
   local +z point world-down. The OmniLidar child prims were spawned with
   identity orientation, so the emitter elevation angles were authored in a
   frame whose +z=world-down — rays at positive elevation went up in sensor
   frame = down in world = into the floor below blind-radius. Fixed:
   `spawn_mid360` called with `orientation=MID360_QUAT_WXYZ` (180° roll) so
   the prim's frame matches the physical mounting convention.

**Prim validation test**: `scripts/test_lidar_prim.py` — all 4 prims PASS
(uv venv, headless, confirmed 2026-08-13):
- `elementsCoordsType = CARTESIAN` ✅
- `accumulateOutputs = True` ✅
- `tickRate = scanRateBaseHz = 10.0` ✅
- s000 elevation = `[-7.2°, +52.2°]` on all prims ✅
- orientation = 180° roll (w≈0, x≈1) ✅

**Standalone visualizer**: `scripts/lidar_viz_isaacsim.py` — flat Mesh ground
+ sensor at configurable height, draws points in viewport with elevation
colour, prints per-prim diagnostics. Run this first to confirm sensor sees
geometry before introducing the robot/ROS2 stack.

**Not yet re-run on real warehouse**: fixes applied and prim attrs verified.
Next: run `python scripts/lidar_viz_isaacsim.py` (flat ground, no robot) to
confirm hits, then re-run full `g1_warehouse_sim.py` and check the published
topics.

## Locomotion — untested edges

- ⬜ Turning/strafing (`vy`/`wz`) — only forward (`vx=0.5`) exercised live;
  structurally identical in the observation contract but unverified.
- ⬜ 50 Hz-trained vs. 60 Hz-physics control-rate mismatch — stable so far,
  not an exact reproduction of training conditions.
- ⬜ Lateral drift — MuJoCo validation showed ~0.68 m sideways drift over
  17 s of a pure-forward command; genuine gait asymmetry vs. harness
  artifact is unknown.
- ⬜ Longest live Isaac Sim run so far is ~15 s sim-time; no multi-minute
  soak test yet.

## Warehouse SDG pipeline (3-process design)

- ✅ **Process 1 (sim)** — `g1_warehouse_sim.py`, verified.
- 🔄 **Process 2 (`scripts/g1_patrol.py`)** — standalone open-loop
  `/g1/cmd_vel` polygon-patrol commander, written, never run together with
  process 1. Dead-reckoning only, no localization — will drift and not
  close its loop by design; real odometry/SLAM would be needed for a
  closed-loop version.
- ⬜ **Process 3 (rosbag capture)** — capture command documented, full
  3-minute run not yet executed, bag→video script not yet built
  (`cv_bridge` + OpenCV `VideoWriter` available in the ROS2 Jazzy env, just
  not wired up).

## Detection

- ✅ CenterPoint ported to torch 2.7/CUDA 12.6 (`detection/livox_centerpoint.py`),
  validated against synthetic scene (see Plan.md).
- ✅ Tested against the RTX sensor's real published cloud — runs, 0
  detections, expected (targets were inside the blind radius / outside the
  covered band at the time).
- ⬜ Not yet tried against a populated warehouse scene with IRA humans.
- ⬜ **OpenPCDet PointPillar integration** — real pretrained weights
  confirmed working standalone in a separate smoke test; not wired into
  this pipeline. See Plan.md for the two scoped integration paths.
- `g1_perception_ws/` (sibling ROS2 workspace): `lidar_bridge`,
  `detection_bridge`, `ccvnorm_node` (depth completion), `centerpoint_node`,
  `pointpillar_node` all build clean (`colcon build --symlink-install`), ⬜
  never tested against a live sim or real robot.

## SLAM — not integrated

Ultra-Fusion runs correctly via Docker on this machine but has no sensor
profile for our legged/wheel-less robot and no real Mid-360/D435 extrinsics
calibrated yet. See Plan.md for alternatives (FAST-LIO2, FAST-LIVO2, etc.).

## Foxglove Visualization (ACTIVE ISSUE)

- ⛔ **Foxglove web not visualizing data** — Bridge runs on spark2 port 8765,
  topics registered, no errors in log, port accessible from laptop (HTTP 426),
  but Foxglove web (`app.foxglove.dev`) shows nothing when connected to
  `ws://10.131.171.77:8765`. Previous error: "Failed to parse channel schema"
  + "Cannot read properties of undefined (reading 'type')".
- 🔄 **Attempted fixes:**
  1. ROS2 message type names + simple JSON schemas → schema parse errors
  2. Protobuf encoding with foxglove.FrameTransforms/PointCloud → "no such type"
  3. Complete JSON schemas + foxglove-native schema names → bridge OK, Foxglove empty
- ⬜ **Next attempts:**
  1. Try `rosbridge_server` (ros-jazzy-rosbridge-suite) inside Docker container
  2. Try `foxglove-sdk` Python package (replacement for deprecated foxglove-websocket)
  3. Try desktop Foxglove app instead of web
  4. Capture Foxglove browser console logs for exact error details
  5. Try omitting `schemaEncoding` parameter entirely

## Infra notes

- GPU is 8 GB — only one Isaac Sim instance fits. Kill stragglers before
  every run: `pkill -9 -f g1_warehouse_sim`.
- Disk is tight machine-wide (~5 GB free / 355 GB at last check, mostly
  unrelated causes) — check `df -h /` before any large asset/Docker pull.
- GEAR-SONIC (`gear_sonic_deploy`) is parked, not finished — TensorRT 10.13
  installed, checkout untouched under
  `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/`, in case its
  fuller manipulation+locomotion capability is wanted later.
- `record_g1_bag.sh` targets **real-hardware** topics (`/utlidar/cloud`,
  `/g1/odom`), neither of which this sim publishes — don't reach for it
  when testing the sim pipeline; use a raw `ros2 bag record` command
  instead (topic list: `/livox/mid360/points`, `/g1/camera/*`, `/tf`,
  `/g1/joint_states`, `/g1/imu`, `/clock`).
