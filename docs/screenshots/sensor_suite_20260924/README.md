# Sensor suite test — 2026-09-24

Headless warehouse sim, Dex3 G1, internal WBC, no IRA:

```bash
python -u scripts/g1_warehouse_sim.py --headless --wbc-mode internal --no-ira \
  --capture-dir docs/screenshots/sensor_suite_20260924 --capture-step 900
```

## Results

| Check | Result |
|---|---|
| `scripts/verify_sensor_tf.py` → `verify_sensor_tf.json` | **PASS 11/11**. Lidar floor z −0.003 ± 0.011 m, elevation p1/p99 −6.3…49.2°, 36/36 azimuth bins, ~13.1k pts/scan |
| `scripts/check_sensor_suite.py` → `check_sensor_suite.txt` | **PASS**. All topic rates, 4 IMUs at 9.81 m/s², 43 joints (14 Dex3), `/g1/hand_cmd` and `/g1/arm_cmd` tracking, WBC walk 1.4 m upright |

A verifier run started during robot settle-in (< step 200) failed
`lidar nothing below floor`: floor std was 4.4 cm, compared with 1.1 cm once settled. Run the
verifier after the WBC has settled (≥ ~600 steps).

## Images

| File | Source |
|---|---|
| `isaac_front.png`, `isaac_side.png`, `isaac_overview.png`, `isaac_behind.png` | Isaac Sim RTX, third-person cameras (`CAPTURE_VIEWS` in `g1_warehouse_sim.py`), 1280×720 |
| `isaac_d435_rgb.png`, `isaac_d435_depth.png` | `/g1/camera/rgb` and `/g1/camera/depth` from the sim (depth TURBO, 0–5 m) |
| `rviz_overview.png`, `rviz_side.png`, `rviz_front.png`, `rviz_top.png` | RViz2 `rviz/g1_rtx.rviz`: Mid-360 cloud (red), D435 depth and colour clouds, RobotModel, TF, RGB and depth panels |

The RViz `Mid360_A` and `CameraColorCloud` displays sometimes show `Status: Error`.
This is a transient TF message-filter drop under `use_sim_time`; the clouds still render.

## Findings

- **Steering wheel is missing.** The log says it was added at
  `/World/Props/SteeringWheel`, but it is not visible on the table in any view.
  It is probably a dynamic-body problem, such as a nested rigid body or a missing collider. This
  blocks the GR00T grasp test.
- The robot stands crouched with its forearms raised. This is the existing WBC stance plus the
  elbow −1.2 rad start pose, not a regression.
- Capture gotcha: Isaac's `Articulation._on_prim_deletion` drops its physics
  view on **any** prim deletion, so `render_product.destroy()` mid-run kills
  the WBC loop. The capture code parks one render product
  (`hydra_texture.set_updates_enabled(False)`) and switches cameras via
  `hydra_texture.camera_path`. Retargeting the USD `camera` relationship
  gives blown-out white frames. `g1.step()` must also keep running during the
  capture frames. If it skips them, the lidar publisher falls out of sync and
  publishes about 1.5× points per scan for the rest of the run.
