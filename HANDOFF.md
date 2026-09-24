# HANDOFF — G1 Foundation Model Testing + TF/Camera/Lidar Alignment

**Status**: CHECKPOINT — UI server verified on Spark02; Dex3/TF bridge and UI code added; GR00T N1.7 access test running; full Docker stack pending
**Last Updated**: 2026-09-23
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
   - A GPU load verification is running. Do not substitute Cosmos3-Edge/Nano or
     silently switch to N1.5 if the load fails.

2. **UnifoLM-VLA is the secondary action model.**
   - The Unitree checkpoint download is running on Spark02 at
     `~/foundation_models/UnifoLM-VLA-Base` (about 19 GB).
   - Its WebSocket adapter and full dependency validation are still pending.
   - Keep Cosmos models optional and unloaded by default.

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
- The Spark02 UI process is running and the browser smoke test passed.
- No Docker container has been started yet for the real Isaac Sim + model stack.

### New checkpoint — server, GPU venv, HF access, Dex3/TF work

- Spark02 UI is running again at `http://10.131.37.135:8080`; Playwright/Chrome
  smoke check passed. The offline URDF view remains:
  `http://10.131.37.135:8080/?demo=1&view=3d-view`.
- Spark02 `~/venvs/gr00t` now has `torch 2.14.0+cu130`,
  `torchvision 0.29.0+cu130`, CUDA 13.0, and `torch.cuda.is_available() == True`.
- The current local Hugging Face credential was forwarded to Spark02 through
  SSH stdin only; `hf auth whoami` returned `Thakk100`. The token is not stored
  in the repository, Docker files, or command arguments.
- GR00T N1.7 load verification was started with the forwarded token and the
  GPU venv. Check its process/log before relying on inference; do not start a
  second copy while it is running.
- UnifoLM-VLA-Base download was started on Spark02 at
  `~/foundation_models/UnifoLM-VLA-Base` (approximately 19 GB). Its code/adapter
  is not yet wired into the UI service.
- The Dex3-capable robot USD and contact/TF changes are staged for this
  checkpoint. `g1_robot.py` now defaults to
  `assets/robot/g1_29_dex3/g1_29dof_with_dex3_base_fix.usd`, publishes 43
  joints (29 body + 14 Dex3), and accepts `/g1/hand_cmd`.
- `sim_ws_bridge.py` now serializes `/tf` and `/tf_static`; the UI has a TF tree
  panel that marks Dex3 frames. This still needs a live ROS 2 run test.
- The model selector now includes UnifoLM-VLA, but no UnifoLM inference process
  is active yet. GR00T remains the only implemented model server.
- Model loading is lazy for GR00T (weights load on first inference), but a
  multi-process hot-unload manager is not implemented yet. Do not claim all
  models are hot-loadable until the manager exists.

The older sections below preserve prior experiments; where they conflict with
this checkpoint, use the checkpoint and resume order above.

---

### Resume order

0. Check the one running Spark02 GR00T load test and the UnifoLM download; do not duplicate either.
1. Fix `/tf` and `/tf_static` streaming in `scripts/sim_ws_bridge.py`.
2. Add TF data to the UI input panel.
3. Write the UnifoLM-VLA WebSocket adapter using the Spark02 checkpoint.
4. Add a multi-process model manager with explicit load/unload and idle eviction.
5. Add `gr00t` and `unifolm` services to the Docker Compose stack.
6. Start Isaac Sim + bridge + GR00T/UnifoLM + UI only after the GPU/container environment is verified.
7. Keep WBC on legs/waist and route only the selected arm/hand trajectory after explicit preview approval.

---

## 🎯 TWO PARALLEL WORKSTREAMS

| Workstream | Status | Priority |
|------------|--------|----------|
| **A. TF/Camera/Lidar Alignment** | 🟡 Dex3/TF code added; live ROS 2 validation pending | **HIGH** |
| **B. Foundation Model Testing (GR00T/UnifoLM/Cosmos)** | 🟡 N1.7 GPU access test running; UnifoLM download running; full services pending | **HIGH** |

---

## 📋 WORKSTREAM A: TF/CAMERA/LIDAR (PRIORITY)

### ✅ WORKING

| Component | Status | Evidence |
|-----------|--------|----------|
| **Lidar** (`/livox/mid360/points/a`) | ✅ Fixed | World Z ∈ [-0.04, 3.67], mean 1.74 — floor at z=0 |
| **Color Depth** (`/g1/camera/depth/color/points`) | ✅ Fixed | Optical frame → world Z ∈ [-0.39, 0.64], mean -0.13 |
| **Static TF** `d435_link → d435_color_optical_frame` | ✅ Publishes | Quaternion (-0.5, 0.5, -0.5, 0.5) — REP-103 |
| **Lidar spawn** | ✅ Identity | `spawn_mid360` default = identity (no double 180° roll) |

### ⚠️ Validation still required

The Dex3/TF implementation is now in the repository, but the following
live checks remain:

- Start Isaac Sim and verify `/tf` and `/tf_static` reach the WebSocket bridge.
- Confirm the TF panel shows the Dex3 palm/fingertip frames.
- Re-run the depth/TF alignment check after the Dex3 USD is loaded.
- Do not claim the full simulator input contract is validated until these
  checks pass.

### KEY FILES (TF/Camera)

| File | Purpose |
|------|---------|
| `g1_sim/rtx_lidar.py` | `spawn_mid360` — identity default orientation |
| `g1_sim/g1_robot.py` | Warehouse calls identity orientation |
| `g1_sim/rtx_camera.py` | `attach_camera_publishers()` — OmniGraph with CameraTF |
| `assets/g1_29dof_sensors.usd` | USD with baked transforms |
| `rviz/g1_rtx.rviz` | RViz config (Fixed Frame = World) |

### IMMEDIATE DEBUG TASKS (TF)

1. **Check CameraTF node in OmniGraph**
   - Graph: `/ActionGraph/CameraROS2`
   - Node: `CameraTF` (ROS2PublishRawTransformTree)
   - Must have exec from `OnTick.outputs:tick`

2. **Verify camera prim name in USD**
   ```python
   from pxr import Usd
   stage = Usd.Stage.Open('assets/g1_29dof_sensors.usd')
   for p in stage.Traverse():
       if 'camera' in p.GetName().lower():
           print(p.GetPath())
   ```

3. **Check static TF emission**
   ```bash
   ros2 topic echo /tf_static --once | grep -A 10 "d435_camera"
   ```

4. **Verify depth_pcl frame_id**
   ```bash
   ros2 topic echo /g1/camera/depth/points --once | head -5
   # Should show frame_id: d435_camera
   ```

### EXPECTED CORRECT STATE

| Frame | World Z | Notes |
|-------|---------|-------|
| `mid360_link` | ~1.21 | 0.8 + 0.4188 |
| `d435_link` | ~1.21 | 0.8 + 0.41987 |
| `d435_camera` | ~1.21 | Same as d435_link (fixed offset) |
| `d435_color_optical_frame` | ~1.21 | Same origin, rotated |

---

## 🤖 WORKSTREAM B: FOUNDATION MODEL TESTING (READY TO RESUME)

### Models Available

| Model | Status | Location | Notes |
|-------|--------|----------|-------|
| **GR00T-N1.7-3B** | ✅ Downloaded (6.5 GB) | `~/foundation_models/GR00T-N1.7-3B` on spark02 | `REAL_G1` pretrain tag |
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

# On laptop: start warehouse sim (with table + steering wheel)
cd /home/thakk100/Projects/thesis/G1_sim
source .envrc
python scripts/g1_warehouse_sim.py --wbc-mode internal --no-ira --config-dir assets/lidar_configs_rotary

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
| Sim (main) | `/tmp/opencode/warehouse_gui13.log` |
| Sim (alt) | `~/warehouse_sim.log` |
| WS Bridge | `~/ws_bridge_test.log` |
| GR00T Server | `~/gr00t_ws_server.log` (on spark02) |

---

## 🔧 QUICK FIXES APPLIED (NOT YET COMMITTED)

```bash
# Files modified since last commit:
M  g1_sim/arm_override.py         # spin_once fix (rclpy.spin_once)
M  g1_sim/rtx_publisher.py        # invert_z param + Z flip
M  g1_sim/g1_robot.py             # pass invert_z=True to lidar
M  scripts/g1_warehouse_sim.py    # add_locomanip_props (table + wheel)
M  scripts/ws_sensor_bridge.py    # import fixes (Node, ARM_CMD_TOPIC)
M  scripts/gr00t_ws_server.py     # websockets 17.x compat, max_size, server closure
?? scripts/ws_sensor_bridge.py
?? scripts/gr00t_ws_server.py
```

---

## 📋 NEXT STEPS (IN ORDER)

### 1. Fix Camera TF (Workstream A — DO THIS FIRST)
```bash
cd /home/thakk100/Projects/thesis/G1_sim
source .envrc
# Debug OmniGraph CameraTF node emission
# Fix rtx_camera.py attach_camera_publishers() to emit static TF
```

### 2. Commit Foundation Model Setup (Workstream B)
```bash
git add g1_sim/arm_override.py g1_sim/rtx_publisher.py g1_sim/g1_robot.py \
        scripts/g1_warehouse_sim.py scripts/ws_sensor_bridge.py scripts/gr00t_ws_server.py
git commit -m "feat: GR00T WebSocket bridge + arm override + locomanip props"
```

### 3. Resume Foundation Model Testing
- Verify Camera TF fixed → depth points align with lidar
- Run GR00T bridge with proper RGB (currently frame too large)
- Test Cosmos3-Edge for video prediction / planning
- Integrate steering wheel grasp task

---

## 🧭 MENTAL MODEL: HOW TO THINK ABOUT THIS

| Layer | Responsibility | Owner |
|-------|----------------|-------|
| **WBC (legs/waist)** | Balance, walk, stand | `WbcBridge` (ONNX) — never touches arms |
| **Arm Override** | Hold, reach, grasp | `ArmTargetSubscriber` → `/g1/arm_cmd` |
| **Foundation Model** | Predict arm targets | GR00T/Cosmos on spark → WebSocket → `/g1/arm_cmd` |
| **Sensors** | RGB, depth, lidar, joint_states | Isaac Sim RTX + ROS2 bridge |
| **Props** | Table, steering wheel | USD references in `add_locomanip_props()` |

**Key invariant**: WBC and Foundation Model **never share joints**. WBC owns 15 leg/waist joints; Foundation Model owns 14 arm joints. This decoupling is intentional and working.

---

## 📝 NOTES FOR RESUMPTION

1. **Spark02 IP**: Use `10.131.37.135` (not 10.131.140.170) — check `hostname -I` on spark02
2. **GR00T load time**: ~12s on CPU — wait for `[WS] GR00T arm server on ws://0.0.0.0:8765`
3. **RGB frame size**: 640×480 RGB = ~900KB base64 → need `max_size=10MB` on server
4. **Camera TF fix**: Once fixed, depth pointcloud will align with lidar in RViz
5. **Steering wheel**: Dynamic rigid body on kinematic table — ready for GR00T grasp test

---

**Next agent**: Fix `CameraTF` static TF emission in `g1_sim/rtx_camera.py`. The OmniGraph node exists but doesn't publish to `/tf_static`. Once fixed, resume foundation model testing with aligned sensors.