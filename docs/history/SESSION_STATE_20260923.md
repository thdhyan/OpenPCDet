> **Superseded (2026-09-23):** historical notes. Current state and resume steps live in `HANDOFF.md` (Workstream A) and `docs/TACSL_PLAN.md`.

# Session State — 2026-09-23

## Summary
Continued the G1 foundation-model open-loop test setup. Key achievements: fixed arm override DDS bug, added lidar Z-inversion, wrote WebSocket bridge code, started GR00T inference env on spark02.

## Completed This Session

### 1. Arm Override Channel Fixed (committed `1841a2f`)
- **File**: `g1_sim/arm_override.py`
- **Bug**: `self._node.spin_once()` doesn't exist on `rclpy.Node`
- **Fix**: Changed to `rclpy.spin_once(self._node, timeout_sec=0.0)` matching `joint_cmd_sub.py`/`rtx_publisher.py` pattern
- **Verified**: Sim runs, arm override channel active, "EXTERNAL arm targets active" log fires

### 2. Lidar Z-Axis Inversion (uncommitted)
- **Files**: `g1_sim/rtx_publisher.py`, `g1_sim/g1_robot.py`
- **Change**: Added `invert_z=True` parameter to `RtxLidarPublisher`, flips Z in `_to_message()`
- **Applied**: `handle._lidar_pub = RtxLidarPublisher(..., invert_z=True)` in `g1_robot.py` L367
- **Status**: Compiled OK (`PYCOMPILE_OK`), sim7 booting (not yet at "running")

### 3. WebSocket Bridge (new files, uncommitted)
- **scripts/ws_sensor_bridge.py** — Laptop client:
  - Subscribes `/g1/camera/rgb`, `/g1/joint_states` locally
  - Sends base64 RGB + joint dict over WS to spark
  - Receives `arm_cmd` (14-D ARM_JOINTS) and publishes to `/g1/arm_cmd`
  - Reconnection logic, 5 Hz default
- **scripts/gr00t_ws_server.py** — Spark server:
  - Accepts single WS connection (laptop-initiated)
  - Loads `Gr00tPolicy` with `REAL_G1` pretrain tag
  - Builds observation from laptop data (approximates EEF 9d from arm joints)
  - Returns absolute arm joint targets (integrates RELATIVE deltas)
  - Hold-last semantics (resends last targets on dropout)

### 4. Spark02 Foundation Model Setup
- **Models downloaded** (114 Gi RAM free, 1.9 T disk):
  - `GR00T-N1.7-3B` — 6.5 GB ✓
  - `Cosmos3-Edge` — 8.6 GB ✓
  - `Cosmos-Reason2-2B` — FAILED (gated, token lacks access)
- **Venv**: `~/venvs/gr00t` with torch 2.9.0 CPU (aarch64)
- **GR00T deps installing**: `tyro`, `cryptography`, `gitpython`, `jsonlines`, `onnx`, `onnxscript`, `tensorrt-cu13` — in progress when server restarted

### 5. REAL_G1 Modality Config (from model statistics.json)
```
video: ego_view (T=2, Δ[−20,0])
state (T=1): left_wrist_eef_9d(9), right_wrist_eef_9d(9), left_hand(7), right_hand(7), left_arm(7), right_arm(7), waist(3)
action (T=40): same 7 + base_height_command(1), navigate_command(3) — RELATIVE for arms, ABSOLUTE for hands/waist
language: annotation.human.task_description
```

## Files Changed
```
M  g1_sim/arm_override.py         # spin_once fix
M  g1_sim/rtx_publisher.py        # invert_z param + Z flip
M  g1_sim/g1_robot.py             # pass invert_z=True
?? scripts/ws_sensor_bridge.py    # laptop WS client
?? scripts/gr00t_ws_server.py     # spark WS server
?? cyclonedds_laptop.xml
?? cyclonedds_tcp.xml
?? dl/
?? launch/run_perception.sh
?? mcp_servers/
?? scripts/build_mid360_assets.py
?? scripts/mid360_unitree_g1.usd
```

## Known Issues / Next Steps
1. **TF root cause**: `spawn_mid360` applies 180° roll on child prims, but URDF already bakes it into `mid360_link`. Publisher negates Z as band-aid; fix is `orientation=(1,0,0,0)` in `spawn_mid360` call.
2. **Spark02 GR00T deps**: Need `tyro` + remaining pip installs to complete. Use `--no-deps` editable install after deps.
3. **Sim7**: Still booting (215s elapsed, not at "WH running" marker). RViz4 launching.
4. **Teleop**: Window alive (PID 51145) — survived restarts.
5. **Commit pending**: Lidar invert + bridge files need commit + push.

## Resume Commands
```bash
# On laptop: restart sim + RViz
cd /home/thakk100/Projects/thesis/G1_sim
source .envrc
python scripts/g1_warehouse_sim.py --wbc-mode internal --no-ira --config-dir assets/lidar_configs_rotary

# On spark02: finish GR00T deps
ssh aim_spark02
VG=~/venvs/gr00t
"$VG/bin/pip" install tyro==0.9.17 cryptography gitpython jsonlines onnx onnxscript tensorrt-cu13 tensorrt-cu13-libs
"$VG/bin/pip" install -e ~/Isaac-GR00T --no-deps
"$VG/bin/python" -c "import sys; sys.path.insert(0,'~/Isaac-GR00T'); from gr00t.policy import Gr00tPolicy; print('OK')"

# Test WS bridge (laptop)
python scripts/ws_sensor_bridge.py --host 10.131.140.170 --port 8765 --text-cmd "hold a box"

# Test GR00T server (spark)
python scripts/gr00t_ws_server.py --model ~/foundation_models/GR00T-N1.7-3B --port 8765
```