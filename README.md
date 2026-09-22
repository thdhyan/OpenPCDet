# G1 Sim — Unitree G1 + Livox Mid-360 + D435 + WBC + Warehouse

Isaac Sim 6.1.0 simulation of a Unitree G1 (29 DOF) carrying a Livox Mid-360
LiDAR (RTX prim, default profile spawns 1 → `/livox/mid360/points/a`) and
D435 RGBD camera, standing/walking via GR00T
whole-body control, in a populated warehouse. Publishes over ROS 2 Jazzy.

Design rationale → [Plan.md](Plan.md). Status board → [Tasks.md](Tasks.md).
Run commands / gotchas → [HANDOFF.md](HANDOFF.md).

---

## Environment

| Item | Value |
|---|---|
| Python env | uv venv at `/generalSSD/IsaacLab/isaac6/.venv`, Python 3.12, Isaac Sim 6.1.0 |
| Activated via | `/generalSSD/IsaacLab/isaac6/.envrc` (sourced by `G1_sim/.envrc`) |
| Isaac Lab | 3.0 EA (`release/3.0.0`) checkout at `/generalSSD/IsaacLab-release-3.0.0`, installed editable into the venv |
| ROS 2 | Jazzy — in-sim `rclpy` bundled in the venv (`isaacsim.ros2.core`); `ros2` CLI from system `/opt/ros/jazzy` |
| GPU | RTX 4060 Laptop, 8 GB VRAM — only one sim instance fits |
| torch | 2.12.0+cu130 |

---

## Installation (fresh machine / server box)

Reproduces the environment in the table above from scratch. Official
reference: [Isaac Lab v3.0.0-EA installation](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html) —
the `release/3.0.0` branch targets **Isaac Sim 6.1 + Python 3.12**.

### 1. Prerequisites

- Ubuntu 22.04+ x86_64 (aarch64 works — see server notes), GLIBC ≥ 2.35
- NVIDIA production driver **≥ 580.65.06** (CUDA 13 PyTorch build;
  `≥ 580.142` on DGX Spark)
- ~40 GB free disk (wheels + extension cache), ≥ 32 GB RAM. 16 GB VRAM is
  the nominal recommendation; headless single-sim runs fit in 8 GB
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
- aarch64 / DGX Spark build deps:

  ```bash
  sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev \
     libxi-dev libxinerama-dev libxrandr-dev
  ```

### 2. Isaac Sim 6.1.0 (pip) + PyTorch cu130

```bash
VENV=/generalSSD/IsaacLab/isaac6/.venv   # hard-coded in the .envrc files — change both if different
uv venv --python 3.12 --seed "$VENV"
source "$VENV/bin/activate"
uv pip install --upgrade pip

uv pip install "isaacsim[all,extscache,ros2]==6.1.0.0" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow
uv pip install -U torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130

export OMNI_KIT_ACCEPT_EULA=YES          # required for headless / non-interactive boots
isaacsim --no-window                     # smoke test
```

Extras: `extscache` pre-caches the Kit extensions (fast, near-offline first
boot); `ros2` bundles the in-sim ROS 2 Jazzy `rclpy` + bridge OmniGraph nodes
this project publishes with.

### 3. Isaac Lab 3.0.0 EA (editable checkout)

```bash
git clone --branch release/3.0.0 https://github.com/isaac-sim/IsaacLab.git /generalSSD/IsaacLab-release-3.0.0
cd /generalSSD/IsaacLab-release-3.0.0
source "$VENV/bin/activate"              # same venv as Isaac Sim
sudo apt install cmake build-essential
./isaaclab.sh -i                         # editable core packages + default Newton/RL/visualizer deps
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless   # verify
```

`-i` installs `isaaclab`, `isaaclab_tasks`, `isaaclab_rl`, … editable from
`source/`. Minimal variant: `./isaaclab.sh -i core`, plus e.g.
`-i 'rl[rsl-rl]'` for a single RL library.

### 4. This repo

```bash
git clone <repo-url> G1_sim && cd G1_sim
direnv allow                             # activates G1_sim/.envrc (venv + PYTHONPATH)
rsync -a otherhost:G1_sim/assets/ assets/   # assets/ is git-ignored (robot USD, ONNX policies, lidar configs)
python scripts/g1_warehouse_sim.py --headless   # first run pulls the warehouse USD from S3 CDN
```

### 5. Server-box notes (headless)

- **No display needed** — every entrypoint runs `--headless`; the `.envrc`
  already exports `OMNI_KIT_ACCEPT_EULA=YES`.
- aarch64 (DGX Spark): install the build deps from step 1; if `libgomp`
  warnings appear, prefix Python with
  `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`.
- **Never source `/opt/ros/jazzy` inside the venv shell** — it clashes with
  the bundled `isaacsim.ros2.core` rclpy. Use a second, non-venv shell for
  the `ros2` CLI (see Quickstart).
- **Never `uv sync` in this venv** — it is not a uv project env; a sync
  re-resolves everything and can clobber pinned working versions (e.g.
  `charset_normalizer==3.5.1`). Add packages with `uv pip install <pkg>`.
- Budget VRAM: one sim instance per ~8 GB GPU. Check `nvidia-smi` before
  co-running training + sim on a small card.
- Results are plain files — pull them back over SSH
  (`rsync` `screenshots/`, `logs/`).
- Docker/HPC: the perception container lives in `docker/` (Dockerfile,
  compose, CycloneDDS XMLs). For Isaac Lab in containers use
  `./docker/container.py start`, then convert the image to Apptainer and
  submit with `sbatch` on SLURM clusters — see the
  [Docker and HPC section](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html#docker-and-hpc-clusters).

---

## Quickstart

With [direnv](https://direnv.net/) installed, entering `G1_sim/` auto-activates the env:

```bash
cd ~/Projects/thesis/G1_sim   # direnv activates .envrc automatically
direnv allow                   # first time only — approve the .envrc
```

Without direnv, activate manually:

```bash
source /generalSSD/IsaacLab/isaac6/.envrc
cd ~/Projects/thesis/G1_sim
```

**Terminal 1 — simulation** (warehouse + G1 + WBC + camera + LiDAR, headless):

```bash
python scripts/g1_warehouse_sim.py --headless
```

**Terminal 2 — RViz** (system ROS 2 Jazzy CLI):

```bash
unset GTK_PATH GIO_MODULE_DIR LOCPATH   # required — see Gotchas
source /opt/ros/jazzy/setup.bash
ros2 launch ~/Projects/thesis/G1_sim/launch/g1_bringup.launch.py detection:=false
```

> **Note:** the venv contains **no `ros2` binary** — only the in-sim `rclpy`
> libs (`isaacsim.ros2.core`). The `ros2` CLI comes from the system
> `/opt/ros/jazzy` install, so run it in a shell **without the venv active**
> (direnv only activates on `cd` into `G1_sim/`, hence the absolute launch
> path above — verified working from outside the repo dir). Sourcing
> `/opt/ros/jazzy/setup.bash` in the same shell as the venv conflicts with
> the bundled rclpy (see the parent `.envrc`).

---

## Directory layout

```
G1_sim/
├── scripts/
│   ├── g1_warehouse_sim.py        # main entrypoint — sim + all sensors + WBC
│   ├── g1_rtx_sim.py              # bare warehouse-free alternative scene
│   ├── g1_patrol.py               # open-loop /g1/cmd_vel patrol commander
│   ├── gen_mid360_rtx_config.py   # generates emitter-state JSON configs
│   ├── convert_g1_urdf_to_usd.py  # URDF → USD (offline, no GPU)
│   ├── build_mid360_assets.py     # CSV → emitter-state JSON
│   ├── capture_screenshots.py     # thesis RGB/depth/env shots + LiDAR debug overlay
│   └── bake_mid360_into_usd.py    # bake MID360 OmniLidar into robot USD (+verify)
├── dl/                            # offline USD-build workflow (no SimulationApp)
│   ├── build_mid360_unitree_g1.py # pure pxr USD builder (no GPU/renderer)
│   ├── launch_isaac_sim.py        # headless launcher for dl/ scene
│   ├── mid360_unitree_g1.usd      # binary USD output from build script
│   ├── mid360_unitree_g1.usda     # text USDA output (human-readable)
│   ├── mid360_emitter_states.json # 4 beam groups × 40 states × 1250 emitters
│   └── g1+Cam+warehouse.usda      # combined G1 + camera + warehouse scene
├── g1_sim/                        # sim Python modules
│   ├── rtx_publisher.py           # RtxLidarPublisher — LiDAR → /livox/mid360/points/{a,b,…}
│   ├── rtx_camera.py              # D435 camera + IMU spawn/attach
│   ├── rgbd_publisher.py          # depth → /g1/camera/depth/color/points
│   ├── wbc_bridge.py              # GR00T decoupled_wbc ONNX bridge
│   ├── ira_actors.py              # IRA wandering humans + Nova Carters
│   ├── warehouse.py               # warehouse USD loader
│   └── action_graph.py            # OmniGraph helper
├── detection/
│   ├── livox_centerpoint.py       # CenterPoint ported to torch 2.7/CUDA 12.6
│   ├── detection_node.py          # ROS 2 detection node
│   └── backend.py                 # ClusteringBackend fallback (no GPU/weights)
├── assets/                        # git-ignored — copy manually between machines
│   ├── g1_29dof_sensors.usd       # G1 USD with sensor mount links
│   ├── robot/g1_29/               # URDF + meshes
│   ├── policy/                    # GR00T-WholeBodyControl-{Balance,Walk}.onnx
│   ├── lidar_configs/             # Livox_Mid360_{A,B,C,D}.json
│   └── ira_warehouse_config.yaml
├── launch/
│   ├── g1_bringup.launch.py       # RViz + robot_state_publisher
│   ├── perception_launch.py       # CUVSLAM + NVBLOX (Isaac ROS container)
│   └── tf_fallback.py             # identity map→pelvis TF on /tf for NVBLOX
├── rviz/
│   └── g1_rtx.rviz                # RGB + LiDAR PointCloud2 + robot model + TF
├── docs/
│   ├── RTX_LIDAR_INIT.md          # RTX LiDAR init detail + coverage diagnostics
│   ├── decoupled_wbc_findings.md  # WBC integration bugs and fixes
│   └── ultra_fusion_findings.md   # SLAM alternatives scoped
├── Plan.md                        # design rationale + history
├── Tasks.md                       # status board
└── HANDOFF.md                     # quickstart + gotchas for next session
```

---

## Published topics

| Topic | Type | Source |
|---|---|---|
| `/livox/mid360/points/a` | `PointCloud2` | RTX LiDAR (in-sim rclpy; per-prim suffix `a,b,c,…` — default rotary profile spawns 1 prim) |
| `/g1/camera/rgb` | `Image` | D435 RGB |
| `/g1/camera/depth` | `Image` | D435 depth |
| `/g1/camera/semantic` | `Image` | D435 semantic |
| `/g1/camera/camera_info` | `CameraInfo` | D435 |
| `/g1/camera/depth/color/points` | `PointCloud2` | synthesized RGBD |
| `/tf` | `TFMessage` | OmniGraph |
| `/g1/joint_states` | `JointState` | OmniGraph |
| `/g1/imu` | `Imu` | OmniGraph |
| `/clock` | `Clock` | OmniGraph |

Subscribed: `/g1/cmd_vel` (`Twist`) → WBC velocity command at 50 Hz.

---

## Offline USD build (`dl/`)

For building/inspecting the USD without a GPU or Isaac Sim runtime:

```bash
# one-time: convert Livox CSV to emitter-state JSON
python dl/build_mid360_assets.py --csv /path/to/mid360-real-centr.csv

# build USD (pure pxr, no GPU required)
python dl/build_mid360_unitree_g1.py
# outputs: dl/mid360_unitree_g1.usd + dl/mid360_unitree_g1.usda
```

The `.usda` (text) is human/AI-readable. The `.usd` (binary) is fast-loading
in Isaac Sim GUI. Both are produced by one `Export()` call — extension
determines format.

**Architecture of the built USD:**

```
/World/G1                        ← Xform → G1 S3 reference
  torso_link/mid360_link/
    Livox_Mid360_{A,B,C,D}       ← OmniLidar prims, 40 emitter states each
  torso_link/d435_link/
    D435/RGB                     ← UsdGeom.Camera (D435)
/World/ground_plane              ← physics plane
/World/ROS2Bridge/               ← OmniGraph action graphs
  ClockGraph → /clock
  TFGraph → /tf, /tf_static
  JointStateGraph → /joint_states
  JointCommandGraph ← /joint_command
  Lidar_{A,B,C,D} → /livox/mid360/{a,b,c,d}
  D435_RGB → /d435/rgb
  D435_Depth → /d435/depth_image
```

**Key schema caveat (Isaac Sim 6.0/6.1):** many attributes from 5.x references
(`tickRate`, `scanType`, `numLines`, `accumulateOutputs`, `instantLidar`, …)
do not exist. Only `skipDroppingInvalidPoints` is a valid
`omni:sensor:Core:*` bool. `tick_rate` is a `Lidar()` constructor param
(`omni:sensor:tickRate`). See `scripts/PROGRESS.md` for the full verified
attribute list.

---

## Sim flags

| Flag | Effect |
|---|---|
| `--headless` | headless (no GUI) |
| `--no-ira` | warehouse only, no IRA dynamic actors |
| `--no-camera` | skip D435 |
| `--no-locomotion` | disable WBC policy |
| `--freeze-robot` | upright pose, gravity off (sensor testing) |
| `--num-prims N` | fewer LiDAR prims |
| `--cache-scene` | (default on) reuse baked warehouse USD |
| `--rebake` | force fresh IRA navmesh bake (~60–80 s) |

---

## Open issues

- **RTX LiDAR coverage partial — root cause identified, fix applied** —
  `GenericModelOutput` default `elementsCoordsType=SPHERICAL` means `gmo.x/y/z`
  are azimuth(deg)/elevation(deg)/range(m), **not** Cartesian. The publisher
  was treating them as Cartesian x/y/z, making elevation appear as a metric
  position (hence the +11°…+88° "band" — those degree values became metre
  offsets). Fix: `spawn_mid360` now sets `omni:sensor:Core:elementsCoordsType="CARTESIAN"`
  on every prim; `RtxLidarPublisher._gather_one` has a fallback SPHERICAL→CARTESIAN
  conversion and runtime check. Prims A/D returning 0 points is a separate
  issue (possibly prim count limit or profile split); still open.
  See `Tasks.md` and `docs/RTX_LIDAR_INIT.md`.
- **Detection untested at scale** — CenterPoint ported and validated on
  synthetic scene; not run against live warehouse + IRA humans.
- **SLAM not integrated** — Ultra-Fusion runs via Docker but needs a sensor
  profile for a wheel-less legged robot. Alternatives: FAST-LIO2, FAST-LIVO2.

---

## Key bugs fixed (history)

| Date | Bug | Fix |
|---|---|---|
| 2026-09-22 | Sporadic GMO `Invalid magic number` in live runs (run7/run8) | Origin inside a mesh, not (only) the #685 race: MID360 mount dropped 2 cm (local z `-0.05`→`-0.03`; flipped frame = world +5→+3 cm) in warehouse/capture/bake |
| 2026-08-13 | LiDAR partial band (+11°…+88° only, no ground returns) | `GMO elementsCoordsType=SPHERICAL` → x/y/z are az/el/range degrees, not Cartesian. Set `omni:sensor:Core:elementsCoordsType="CARTESIAN"` in `spawn_mid360`; added spherical→Cartesian fallback in publisher |
| 2026-08-13 | Intensity always constant 100 | `gmo.scalar` is real normalised intensity (GMO RST docs); publisher now uses it |
| 2026-08-11 | LiDAR cloud published below floor | Mounted sensor on `mid360_link` with identity transform; removed hand-authored 180° roll that double-applied TF |
| 2026-08-10 | Warehouse Nucleus path never resolved | Fixed catalog-relative path in `g1_sim/warehouse.py` |
| 2026-08-10 | IRA navmesh bake always failed at 100-frame cap | Patched poll budget to 3000 frames in `g1_sim/ira_actors.py` |
| 2026-08-10 | G1 collapsed in ~2 s under WBC | Called `Articulation.set_gains()` with real per-joint training gains (kp up to 250) instead of uniform kp=100 placeholder |
