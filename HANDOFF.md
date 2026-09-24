# HANDOFF — G1 Foundation Model Testing + TF/Camera/Lidar Alignment

**Status**: CHECKPOINT — Workstream A (sensors/TF/Dex3) DONE and verified in sim (`9a40ebb`); UI server and lazy GR00T N1.7 GPU service verified on Spark02; UnifoLM assets/dependencies ready but GPU load retry is blocked by another active workload; full Docker stack pending
**Last Updated**: 2026-09-24
**Repo**: `/home/thakk100/Projects/thesis/G1_sim`

---

## CHECKPOINT — 2026-09-24/25 — PAUSE BEFORE TRAVEL

**Decision: keep the Docker-based deployment plan. Do not move the simulator or model server back to the laptop.**

### Target machines

| Service | Target | Reason |
|---|---|---|
| Isaac Sim + RGB/depth/TF/joint bridge | **`dl`** | x86 Docker/Isaac image, 4× RTX 6000 Ada, 503 GiB RAM |
| GR00T N1.7 / UnifoLM GPU servers | **`spark02`** | GB10 GPU; venv now has CUDA-enabled Torch |
| Foundation Model Debug UI | **`spark02`** with SSH/VPN port-forward if needed | Keeps model processes and UI together |
| Spark04 | **Do not use for model storage** | Less than 500 GB free, violates fleet storage rule |

Use port `8080` from the laptop through an SSH tunnel to the UI host:

```bash
ssh -N -L 8080:127.0.0.1:8080 aim_spark02
# open http://localhost:8080
```

The direct Spark02 URL is `http://10.131.37.135:8080/?demo=1&view=3d-view`.
The UI server proxies internally to model `:8765` and simulator bridge `:8766`.

### Model decision

1. **Keep GR00T N1.7 as the active model target.**
   - `GR00T-N1.7-3B` hard-codes `nvidia/Cosmos-Reason2-2B` as its VLM backbone.
   - That repository is gated; the local Hugging Face token was forwarded to
     Spark02 through SSH stdin and authenticated as `Thakk100`.
   - GR00T N1.7 now loads on Spark02 GPU in ~12–15 seconds using
     Torch 2.9.0+cu130; the service falls back to SDPA because the available
     aarch64 FlashAttention wheel is CUDA-12-linked.
   - The background WebSocket service listens on `:8765`; explicit `load`,
     synthetic 43-joint inference, and 40-step preview output were verified.

2. **UnifoLM-VLA is the secondary action model.**
   - The Unitree action checkpoint is present on Spark02 at
     `~/foundation_models/UnifoLM-VLA-Base` (about 19 GB).
   - The required `UnifoLM-VLM-Base` download is in progress.
   - `scripts/unifolm_ws_server.py` is now a lazy EEF23 WebSocket adapter; it
     refuses joint-only input and never dispatches EEF actions as joints. It uses
  SDPA by default on Spark/CUDA 13 because Unitree's hard-coded FlashAttention
  wheel is CUDA-12-linked.
   - Full dependency/model-load validation is still pending.

3. **The old N1.5 locomanipulation checkout is historical fallback material only.**
   - Do not use it as the default or claim it is a drop-in N1.7 replacement.

### Docker services still to finish

The repository contains the compose scaffold, but the complete Isaac Sim +
GR00T/UnifoLM stack is **not yet running**. Keep the split deployment:

```text
dl
└── isaac-sim       :8766 bridge source, GPU

aim_spark02
├── gr00t           :8765, GR00T N1.7, GPU
├── unifolm         :8767, UnifoLM-VLA, GPU
├── sim-ws client   ROS2 -> WebSocket -> Spark02
└── fm-ui           :8080, browser UI and WS proxy
```

Before starting the containers, confirm:

```bash
ssh dl
df -h /
nvidia-smi
docker ps

ssh aim_spark02
df -h /
free -h
nvidia-smi
```

### Simulator stream contract

`sim_ws_bridge.py` now streams:

- `/g1/camera/rgb`
- `/g1/camera/depth`
- `/g1/joint_states`
- `/tf`
- `/tf_static`

Joint states are the primary model state. TF is included as supplementary
state and is displayed in the UI's Dex3-aware TF tree. The new TF path still
needs a live ROS 2 run test.

The warehouse scene contains the robot and the locomanip props added by `add_locomanip_props()`:

```text
/World/Props/PackingTable
/World/Props/SteeringWheel
```

Do not overwrite or regenerate the warehouse USD/USDA until the scene has been visually checked with the table, wheel, robot, RGB, depth, and TF stream in the same run.

### Current Git/UI state

- UI/URDF implementation is in commits `06e96bc` and `9a40ebb`.
- GitHub branch: `isaacsim6-rtx-emitter` on configured `origin`.
- Spark02 clone: `~/Projects/fm-debug-ui`; update it after this checkpoint is
  pushed.
- Verified offline URDF preview URL:
  `http://10.131.37.135:8080/?demo=1&view=3d-view`
- The Spark02 UI process is running (PID/log in `~/fm-ui.pid` and
  `~/fm-debug-ui.log`) and the browser smoke test passed.
- No Docker container has been started yet for the real Isaac Sim + model stack.

### New checkpoint — server, GPU venv, HF access, Dex3/TF work

- Spark02 UI is running again at `http://10.131.37.135:8080`; Playwright/Chrome
  smoke check passed. The offline URDF view remains:
  `http://10.131.37.135:8080/?demo=1&view=3d-view`.
- Spark02 `~/venvs/gr00t` now has `torch 2.9.0+cu130`,
  `torchvision 0.24.0`, CUDA 13.0, and `torch.cuda.is_available() == True`.
- The current local Hugging Face credential was forwarded to Spark02 through
  SSH stdin only; `hf auth whoami` returned `Thakk100`. The token is not stored
  in the repository, Docker files, or command arguments.
- GR00T N1.7 is live on Spark02 at `:8765` (PID/log are in
  `~/gr00t_ws_server.pid` and `~/gr00t_ws_server.log`). The service loaded in
  ~12–15s, fell back to SDPA because the available aarch64 FlashAttention wheel
  is CUDA-12-linked, and returned a verified 40-step synthetic inference preview.
- The UnifoLM-VLA action checkpoint and 16+ GB `UnifoLM-VLM-Base` companion are
  downloaded on Spark02. The official dependencies and editable package are
  installed; `baseframework` and `qwen_vl_utils` import successfully.
- The first UnifoLM GPU load was stopped after the VLM shards loaded because
  Spark's GPU was occupied by another active Ollama `qwen3.6:27b` process. No
  UnifoLM model error was observed; retry the load after that workload exits.
- `scripts/unifolm_ws_server.py` now provides a lazy, unloadable EEF23 adapter.
  It requires an explicit 23-D EEF/base state and intentionally does not guess
  EEF poses from joint angles.
- GR00T's observation builder now passes the 14 Dex3 hand joints from
  `/g1/joint_states` into its `left_hand`/`right_hand` model inputs; the joint
  contract test covers this.
- The Dex3-capable robot USD and contact/TF changes are staged for this
  checkpoint. `g1_robot.py` now defaults to
  `assets/robot/g1_29_dex3/g1_29dof_with_dex3_base_fix.usd`, publishes 43
  joints (29 body + 14 Dex3), and accepts `/g1/hand_cmd`.
- `sim_ws_bridge.py` now serializes `/tf` and `/tf_static`; the UI has a TF tree
  panel that marks Dex3 frames. This still needs a live ROS 2 run test.
- The model selector now includes UnifoLM-VLA, but its inference process is not
  active yet. GR00T is live; UnifoLM remains pending VLM/dependency validation.
- Model loading is lazy for GR00T and UnifoLM's adapter. Both WebSocket
  services accept explicit `{"type":"unload"}` and release weights without
  restarting their listeners; a cross-model supervisor and automatic idle
  eviction are still pending.

The older sections below preserve prior experiments; where they conflict with
this checkpoint, use the checkpoint and resume order above.

---

### Resume order

0. Let the active Spark GPU workload finish; do not launch a second UnifoLM
   load while another process owns the GPU.
1. Run one UnifoLM EEF23 adapter smoke request with an explicit 23-D state.
2. Add a multi-process model manager with automatic idle eviction.
3. Add `gr00t` and `unifolm` services to the Docker Compose stack.
4. Start Isaac Sim + bridge + GR00T/UnifoLM + UI only after the GPU/container
   environment is verified.
5. Keep WBC on legs/waist and route only joint-space arm/hand trajectories after
   explicit preview approval. UnifoLM EEF output remains non-dispatching until
   a separately validated EEF-to-IK stage exists.

---

## 🎯 TWO PARALLEL WORKSTREAMS

| Workstream | Status | Priority |
|------------|--------|----------|
| **A. TF/Camera/Lidar + Dex3** | ✅ Verified live (verify_sensor_tf 11/11, check_sensor_suite PASS, test_dex3_contacts PASS); next = TacSL (`docs/TACSL_PLAN.md`) | MEDIUM |
| **B. Foundation Model Testing (GR00T/UnifoLM/Cosmos)** | 🟡 GR00T live; UnifoLM assets/deps ready, GPU load retry blocked by active Ollama workload | **HIGH** |

---

## 📋 WORKSTREAM A: SENSORS / TF / DEX3 — DONE (2026-09-23)

Everything below was verified live in the headless warehouse sim. Frames are
recorded in `docs/tf_snapshot_20260923.yaml`; the verifier report is
`docs/verify_sensor_tf_20260923.json`. **Do not change the robot loading /
sensor mounts in `g1_sim/g1_robot.py`, `rtx_camera.py`, `rtx_lidar.py`
without re-running the two check scripts** (the code carries DO NOT CHANGE
notes).

### Verify (sim running, `source /opt/ros/jazzy/setup.bash`)

```bash
python3 scripts/verify_sensor_tf.py      # 11 checks: lidar/depth vs TF, floor, FOV
python3 scripts/check_sensor_suite.py    # rates, IMUs, 43 joints, /g1/hand_cmd, /g1/arm_cmd, WBC walk
python scripts/test_dex3_contacts.py     # headless (no running sim): 6 fingertip sensors
```

### What is true now

| Area | State |
|---|---|
| Robot | G1 29-DoF + Dex3 (`assets/robot/g1_29_dex3/g1_29dof_with_dex3_base_fix.usd`, URL in `g1_robot.DEX3_USD_URL`). `load_g1` deactivates the asset's world-weld `root_joint` (floating base for WBC), 43 DOF, spawn yaw +90° facing the packing table (IsaacLab locomanip pose). |
| Mid-360 | Real robot is **inverted**. `load_g1` re-authors `mid360_joint` from current unitree_ros rev 1.0 URDF (`xyz 0.0002835 0.00003 0.428434`, `rpy π 0.0511 0`); the asset had a stale upright joint. RTX prim = identity, +15 cm (`lidar_translation=(0,0,-0.15)` in the inverted link); lower and the head mesh eats the downward rays. |
| Lidar pattern | Real non-repetitive Mid-360: `assets/lidar_configs_solid` (default `--config-dir`) = ONE solid-state emitter state (20k real rays, 100 scan lines); `rtx_lidar.ScanPatternCycler` rewrites it with the next of 40 recorded frames every scan. The engine squeezes elevation with >1 state (10 states → −20…10°) and sweeps azimuth clockwise (author −az). `minReflectance 0.02` keeps grazing floor returns. Regenerate: `python scripts/gen_mid360_solid_config.py`. |
| Lidar numbers | floor z −0.003 ± 0.010 m, elevation p1/p99 −6.3…49.2°, 36/36 azimuth bins, ~13k pts/scan @ 10 Hz, 0 points above 4 m. |
| D435 | Prim at `D435_POS` (URDF + 10 cm fwd/up clearance); optical static TF measured from the stage (`rtx_camera.optical_pose_in_link`), vertical aperture follows 640×480, RGBD cloud uses fy = fx. Depth clouds on floor within ~1 cm, depth vs color 0.6 cm median. |
| ROS | `/g1/joint_states` pos/vel/effort ×43; `/tf` incl. 18 Dex3 frames; `/robot_description` = `assets/robot/g1_29/g1_29dof_with_hand_rev_1_0.urdf` latched by the sim; `/g1/hand_cmd` + `/g1/arm_cmd` (`JointState`, hold-last); `/g1/dex3/<side>/<finger>/contact` (`WrenchStamped`, 60 Hz); 4 IMUs @ 60 Hz. |
| RViz | `rviz/g1_rtx.rviz`: single `Mid360_A`, depth + colorized clouds, RGB/depth images (semantic 32SC1 display removed). Startup shows a transient TF error on the clouds; clears after first TF. |

### Key files

| File | Purpose |
|---|---|
| `g1_sim/g1_robot.py` | `load_g1`: Dex3 un-weld, `_align_mid360_mount`, sensors, `/robot_description` |
| `g1_sim/rtx_lidar.py` | `spawn_mid360`, `ScanPatternCycler`, `pattern_frame_deg` |
| `g1_sim/rtx_publisher.py` | lidar GMO → PointCloud2 in `mid360_link` (prim pose applied) |
| `g1_sim/rtx_camera.py` | D435 prim, camera OmniGraph, measured optical TF |
| `g1_sim/dex3_contacts.py` | fingertip `IsaacContactSensor` + WrenchStamped publisher |
| `g1_sim/arm_override.py` | `/g1/arm_cmd` and `/g1/hand_cmd` (`DEX3_HAND_JOINTS`) subscribers |
| `scripts/unifolm_ws_server.py` | lazy Unitree UnifoLM EEF23 WebSocket adapter; preview-only |
| `scripts/test_unifolm_contract.py` | dependency-free adapter contract check |
| `docs/TACSL_PLAN.md` | tactile roadmap (Phase 0 done) |

### Gotchas that cost hours

- The committed code used to roll the lidar twice → sensor upright → the old
  "360° ring" in `screenshots/run1` was the **ceiling**.
- Legacy scripts still author that double roll (`capture_screenshots.py`,
  `bake_mid360_into_usd.py`, `test_lidar_prim.py` via `MID360_QUAT_WXYZ`).
- Dex3 asset disables self-collision: fingertip contacts read 0 unless an
  external object touches them.
- Memory: the Dex3 warehouse sim is ~7–8 GB RSS. On 2026-09-23 systemd-oomd
  killed the whole terminal scope (tmux + agent). A memory-capped systemd
  unit starved it instead (RTX "Allocation failed" + fd limit 1024). Run in
  tmux with other heavy apps closed.

---

## 🤖 WORKSTREAM B: FOUNDATION MODEL TESTING (READY TO RESUME)

### Models Available

| Model | Status | Location | Notes |
|-------|--------|----------|-------|
| **GR00T-N1.7-3B** | ✅ GPU load + synthetic inference | `~/foundation_models/GR00T-N1.7-3B` on spark02 | `REAL_G1`, lazy WebSocket `:8765` |
| **UnifoLM-VLA-Base** | ✅ Action checkpoint; VLM companion downloading | `~/foundation_models/UnifoLM-VLA-Base` on spark02 | EEF23 preview; not yet live |
| **Cosmos3-Edge** | ✅ Downloaded (8.6 GB) | `~/foundation_models/Cosmos3-Edge` on spark02 | Video/world model |
| **Cosmos-Reason2-2B** | 🟡 Access check in progress | Hugging Face cache on Spark02 | `HF_TOKEN` authenticated; do not assume model load until the test completes |
| **Qwen3.8-Flash-Next** | 🟡 In progress | spark01, llama.cpp branch `qwen4exp` | Separate OpenAI-compatible server on :8001 |

### GR00T Modality Config (REAL_G1 pretrain)

```yaml
video: ego_view (T=2, Δ[-20,0])
state (T=1):
  left_wrist_eef_9d (9), right_wrist_eef_9d (9)
  left_hand (7), right_hand (7)
  left_arm (7), right_arm (7), waist (3)
action (T=40): same 7 + base_height_command (1), navigate_command (3)
  # RELATIVE for arms, ABSOLUTE for hands/waist
language: annotation.human.task_description
```

### Architecture: WebSocket Bridge (Laptop ↔ Spark02)

```
┌─────────────────────────────────────────────────────────────────┐
│  LAPTOP (Isaac Sim)                          SPARK02 (GPU)    │
│  ┌──────────────────┐      WebSocket (ws://)      ┌──────────┐ │
│  │ ws_sensor_bridge │◄────────────────────────────►│ gr00t_   │ │
│  │  (client)        │   JSON frames, ~5 Hz        │ ws_server│ │
│  └────────┬─────────┘                              └────┬─────┘ │
│           │                                           │       │
│  ┌────────▼─────────┐                        ┌────────▼────┐  │
│  │ ROS2 (localhost) │                        │ GR00T Policy│  │
│  │ /g1/camera/rgb   │                        │ (REAL_G1)   │  │
│  │ /g1/joint_states │                        │ 40-step act │  │
│  │ /g1/arm_cmd ◄────┼──── arm_cmd (14-D) ────┤             │  │
│  └──────────────────┘                        └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Frame format** (JSON text frames):
```json
// Laptop → Spark (obs)
{"type":"obs", "t":123.4, "rgb":"base64...", "joints":{"name":[...],"position":[...]}, "cmd":"pick up the steering wheel"}

// Spark → Laptop (arm_cmd)
{"type":"arm_cmd", "t":123.5, "name":[14 ARM_JOINTS], "position":[14 floats]}
```

### Files Created/Modified (Foundation Models)

| File | Purpose |
|------|---------|
| `scripts/ws_sensor_bridge.py` | Laptop WS client: subscribes `/g1/camera/rgb`, `/g1/joint_states`; publishes `/g1/arm_cmd` |
| `scripts/gr00t_ws_server.py` | Spark WS server: loads Gr00tPolicy, returns absolute arm targets |
| `g1_sim/arm_override.py` | ArmTargetSubscriber: buffers `/g1/arm_cmd` for warehouse loop |
| `scripts/g1_warehouse_sim.py` | Added `add_locomanip_props()` — packing table + steering wheel |

### Spark02 Environment

```bash
# GR00T venv
~/venvs/gr00t/bin/python
# Dependencies installed: tyro, cryptography, gitpython, jsonlines, onnx, onnxscript, tensorrt-cu13, websockets
# Isaac-GR00T installed editable: ~/Isaac-GR00T -> /home/thakk100/foundation_models/Isaac-GR00T
```

### Resume Commands (Foundation Models)

```bash
# On spark02: start GR00T WebSocket server
ssh aim_spark02
pkill -9 -f gr00t_ws_server  # clean up any old
nohup ~/venvs/gr00t/bin/python ~/gr00t_ws_server.py \
  --model ~/foundation_models/GR00T-N1.7-3B --port 8765 \
  > ~/gr00t_ws_server.log 2>&1 &

# On laptop: start warehouse sim (Dex3 G1, table + steering wheel, Mid-360 solid pattern by default)
cd /home/thakk100/Projects/thesis/G1_sim
tmux new-session -d -s g1sim "bash -c 'set -a; source .envrc; set +a; python -u scripts/g1_warehouse_sim.py --headless --wbc-mode internal --no-ira > /tmp/opencode/warehouse.log 2>&1'"
# RViz (robot_description comes from the sim):
tmux new-session -d -s g1rviz "bash -c 'source /opt/ros/jazzy/setup.bash; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0; rviz2 -d rviz/g1_rtx.rviz --ros-args -p use_sim_time:=true'"

# On laptop (separate terminal): start WebSocket sensor bridge
cd /home/thakk100/Projects/thesis/G1_sim
source .envrc
python scripts/ws_sensor_bridge.py \
  --host 10.131.37.135 --port 8765 \
  --text-cmd "pick up the steering wheel" --rate 5
```

### Current Bridge Status

- ✅ WebSocket connection establishes (spark02:8765 reachable at 10.131.37.135)
- ✅ GR00T policy loads (3B params on CPU, ~12s load time)
- ⚠️ **Known issue**: RGB frame size exceeds websocket max_size (1MB default)
  - Fix applied: `max_size=10*1024*1024` in `gr00t_ws_server.py`
- ⚠️ **Known issue**: `server` variable captured in closure for handler
  - Fix applied: pass `server` explicitly in `handler_wrapper`

### Arm Override Architecture (Decoupled from WBC)

```
Warehouse Loop (60 Hz):
  ├─ WBC @50Hz → LEG_WAIST_JOINTS (15 DOF) — Balance/Walk policy
  └─ ArmTargetSubscriber → ARM_JOINTS (14 DOF) — GR00T/Cosmos via /g1/arm_cmd
       └─ Hold-last semantics: arms freeze at last target on dropout
```

---

## 🏭 WAREHOUSE SIM — CURRENT STATE

### Running Processes

```bash
# Check tmux sessions
tmux ls
# g1sim    - warehouse sim (running, WBC balanced, step ~8000+)
# g1rviz   - RViz with all pointcloud displays
# g1teleop - keyboard WBC control
```

### Assets Added (Isaac Lab Locomanip Pick-Place)

```python
# Packing Table (kinematic)
prim_path="/World/Props/PackingTable"
usd_path="{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd"
pos=[0.0, 0.55, -0.3]  # y=0.55 forward, z=-0.3 (table top ~0.7m)

# Steering Wheel (dynamic, graspable)
prim_path="/World/Props/SteeringWheel"
usd_path="{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd"
pos=[-0.35, 0.45, 0.6996]  # on table surface
scale=(0.75, 0.75, 0.75)
mass=0.5
```

### Sim Logs

| Log | Path |
|-----|------|
| Sim (main) | `/tmp/opencode/warehouse.log` (last verified run: `/tmp/opencode/warehouse_dex3g.log`) |
| Sim (alt) | `~/warehouse_sim.log` |
| WS Bridge | `~/ws_bridge_test.log` |
| GR00T Server | `~/gr00t_ws_server.log` (on spark02) |

---

## 📋 NEXT STEPS (IN ORDER)

Workstream B resume order is at the top of this file. Workstream A:

1. **TacSL on Dex3** — follow `docs/TACSL_PLAN.md` (Phase 1 feasibility
   spike, then elastomer pads, then the Isaac Lab scene decision).
2. **Hands in the model loop** — route GR00T `left_hand(7)`/`right_hand(7)`
   and the TriHand teleop retargeter output to `/g1/hand_cmd` (per-hand order
   thumb 0–2, index 0–1, middle 0–1 already matches).
3. **Live bridge test** — `sim_ws_bridge.py` `/tf` + `/tf_static` path with
   the Dex3 sim running (the sim side is verified; the bridge side is not).
4. **Legacy scripts** — port `capture_screenshots.py`, `bake_mid360_into_usd.py`,
   `test_lidar_prim.py` to `load_g1` or retire them (double-roll lidar).

---

## 🧭 MENTAL MODEL: HOW TO THINK ABOUT THIS

| Layer | Responsibility | Owner |
|-------|----------------|-------|
| **WBC (legs/waist)** | Balance, walk, stand | `WbcBridge` (ONNX) — never touches arms |
| **Arm / Hand Override** | Hold, reach, grasp | `ArmTargetSubscriber` → `/g1/arm_cmd`, `/g1/hand_cmd` (Dex3) |
| **Foundation Model** | Predict arm targets | GR00T/Cosmos on spark → WebSocket → `/g1/arm_cmd` |
| **Sensors** | RGB, depth, lidar, joint_states | Isaac Sim RTX + ROS2 bridge |
| **Props** | Table, steering wheel | USD references in `add_locomanip_props()` |

**Key invariant**: WBC and Foundation Model **never share joints**. WBC owns 15 leg/waist joints; Foundation Model owns 14 arm joints. This decoupling is intentional and working.

---

## 📝 NOTES FOR RESUMPTION

1. **Spark02 IP**: Use `10.131.37.135` (not 10.131.140.170) — check `hostname -I` on spark02
2. **GR00T load time**: ~12s on CPU — wait for `[WS] GR00T arm server on ws://0.0.0.0:8765`
3. **RGB frame size**: 640×480 RGB = ~900KB base64 → need `max_size=10MB` on server
4. **Sensor TF**: done — see Workstream A; `docs/history/` keeps the old debugging notes
5. **Steering wheel**: Dynamic rigid body on kinematic table — ready for GR00T grasp test

---

**Next agent**: Sensors/TF/Dex3 are verified — start from the checkpoint resume order (Workstream B) or `docs/TACSL_PLAN.md` (tactile). Re-run `scripts/verify_sensor_tf.py` + `scripts/check_sensor_suite.py` after any robot-loading change.