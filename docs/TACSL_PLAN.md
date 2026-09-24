# Dex3 tactile sensing: from tip contacts to TacSL

Status 2026-09-23. Phase 0 is done and verified; later phases are the plan.

## Where we are (Phase 0: simple tip contacts, done)

| Piece | State |
|---|---|
| Robot | Unitree G1 29-DoF + Dex3 hands, `assets/robot/g1_29_dex3/g1_29dof_with_dex3_base_fix.usd` (IsaacLab `G129_CFG_WITH_DEX3_BASE_FIX` asset). `load_g1` removes its world weld (floating base for WBC) and re-authors `mid360_joint` from the current Unitree rev 1.0 URDF (inverted Mid-360). |
| Articulation | 43 DOF: 29 body + 14 Dex3 (`<side>_hand_{thumb_0,thumb_1,thumb_2,index_0,index_1,middle_0,middle_1}_joint`), hand PD kp 8 / kd 1.5 (IsaacLab values). |
| ROS state | `/g1/joint_states` (position, velocity, effort for all 43), `/tf` (all 18 hand frames with their kinematic parents), `/robot_description` = `assets/robot/g1_29/g1_29dof_with_hand_rev_1_0.urdf` latched by the sim. |
| ROS control | `/g1/hand_cmd` (`sensor_msgs/JointState`, any subset of the 14 Dex3 joints, hold-last) - same contract as `/g1/arm_cmd`. |
| Tip contacts | `IsaacContactSensor` on each fingertip link (`thumb_2`, `index_1`, `middle_1` per hand) -> `geometry_msgs/WrenchStamped` on `/g1/dex3/<side>/<finger>/contact` at 60 Hz (sim time); force = net World-frame contact force from raw impulses, torque 0. Code: `g1_sim/dex3_contacts.py`. |
| Verification | `scripts/test_dex3_contacts.py`: all 6 tips report contact against a probe (2-25 N). `scripts/check_sensor_suite.py`: fingers track `/g1/hand_cmd` within 0.014 rad, arms within 0.038 rad, WBC walk + all sensor rates pass. `scripts/verify_sensor_tf.py`: 11/11. |

The tip-contact topics and tip names are the interface TacSL will extend,
not replace: keep `/g1/dex3/<side>/<finger>/contact` (net wrench) and add
tactile arrays beside it.

## What TacSL needs (from `isaaclab_contrib.sensors.tacsl_sensor`)

- `VisuoTactileSensor` is an **Isaac Lab `SensorBase`**: it expects Isaac Lab's
  `SimulationContext` / `InteractiveScene` and PhysX tensor views - not the
  `isaacsim.core.api` loop `scripts/g1_warehouse_sim.py` runs today.
- An **elastomer rigid body** per sensing pad (the sensor prim lives at
  `<elastomer>/tactile_sensor`); tactile points are ray-cast onto the
  elastomer mesh surface, `tactile_array_size=(rows, cols)` with a
  `tactile_margin`.
- **Force-field mode** (penalty normal + shear per taxel) queries the SDF of
  a **contact object declared before sim start**
  (`contact_object_prim_path_expr`, one expression per sensor) whose
  collision mesh has a pre-computed SDF.
- **Camera mode** (GelSight-style RGB) adds one camera + renderer per pad -
  12 pads' worth would not fit next to the RTX lidar/D435 on the 8 GB GPU.
- Reference: `IsaacLab/scripts/demos/sensors/tacsl_sensor.py`
  (GelSight R15 finger, `tactile_array_size=(20, 25)`, margin 3 mm).

## Plan

### Phase 1 - feasibility spike (standalone, no G1)
1. Run the TacSL demo headless, force-field only
   (`--use_tactile_ff`, no RGB), 1 env; record GPU/RAM and step time.
2. Same demo with 6 sensors on one object to estimate the Dex3 budget.
3. Exit criterion: force-field step cost known; decide array size
   (start ~8x6 taxels per fingertip pad, ~2 mm pitch).

### Phase 2 - Dex3 elastomer pads (asset work, USD session layer)
1. Per fingertip, a thin elastomer body on the palmar side of the distal
   link (from the tip collision mesh, or a 20x15x2 mm pad first), fixed-
   jointed to `<side>_hand_<finger>_link`, with its own physics material;
   `tactile_sensor` child prim per TacSL naming.
2. Keep the existing tip `IsaacContactSensor`s (they move to the pad body
   if the pad takes over the contact).
3. Check with `test_dex3_contacts.py` (extended to pads) that pads collide
   and grasp stability is unchanged.

### Phase 3 - runtime architecture (decision point)
- **(A, recommended)** Move the warehouse sim onto Isaac Lab: an env/scene
  cfg mirroring `locomanip_pick_place` (G1 + Dex3, packing table, steering
  wheel) plus our sensors (RTX Mid-360 + cycler, D435, IMUs, ROS graphs).
  TacSL then runs natively and the task matches the Isaac Lab contrib
  locomanip/teleop stack (TriHand retargeters, GR00T data).
- **(B)** Keep `isaacsim.core.api` and drive `VisuoTactileSensor` by hand -
  likely fights its `SimulationContext`/scene assumptions; only if (A) is
  blocked.

### Phase 4 - contact objects
1. Pre-compute SDF collision for the manipulated props (steering wheel,
   crate, ...) and declare them before sim start.
2. One sensor supports one contact-object expression: for several objects,
   either one sensor per (pad, object) or extend the sensor - decide after
   Phase 1 numbers.

### Phase 5 - ROS interface
- Keep `/g1/dex3/<side>/<finger>/contact` (net `WrenchStamped`).
- Add per pad: `/g1/dex3/<side>/<finger>/tactile/normal`
  (`sensor_msgs/Image`, 32FC1, N per taxel) and `.../tactile/shear`
  (32FC2), `frame_id` = new static frame `<side>_hand_<finger>_tactile`
  (pad surface, z out of the pad) published on `/tf_static`.

### Phase 6 - validation
- Known-load press per pad: integrated normal force vs applied (<10 %),
  shear direction under a lateral push, taxel map centered on the contact.
- Rates in sim time next to `check_sensor_suite.py`; GPU/RAM headroom with
  lidar + camera running.

Risks: 8 GB GPU / 15 GB RAM (a Dex3 warehouse run already uses ~7-8 GB RSS;
oomd killed the whole terminal scope once), Isaac Lab develop vs release
3.0 API drift, SDF requirement for every touched object.

## Continue next (outside TacSL)
1. Hand teleop: the pasted TriHand retargeter's 7-joint order
   (`thumb_rotation, thumb_proximal, thumb_distal, index_proximal,
   index_distal, middle_proximal, middle_distal`) matches the per-hand order
   of `DEX3_HAND_JOINTS` (thumb 0-2, index 0-1, middle 0-1) - map it 1:1 onto
   `/g1/hand_cmd`.
2. GR00T REAL_G1 outputs `left_hand(7)` / `right_hand(7)` actions: route them
   to `/g1/hand_cmd` in `scripts/sim_ws_bridge.py` / `gr00t_ws_server.py`.
3. Legacy scripts still author the double 180 deg lidar roll
   (`MID360_QUAT_WXYZ` in `capture_screenshots.py`, `bake_mid360_into_usd.py`,
   `test_lidar_prim.py`) - port them to `load_g1` or drop them.
4. Run the sim with enough free RAM (close browsers/other sims); a memory
   spike took down tmux and the terminal on 2026-09-23.
