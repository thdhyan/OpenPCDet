# Mid360 LiDAR + Unitree G1 + ROS 2 USD — Progress & Plan

Project root: `/home/thakk100/Projects/thesis/`
Conda env: `isaac6` (Isaac Sim 6.0.1 + IsaacLab 3.0.0b2 + IsaacLab Assets)
Goal: Build a single, reusable USD that contains a Unitree G1 with a realistic
Livox Mid360 LiDAR (4 RTX-LiDAR prims, non-repetitive pattern from real CSV data)
plus an Intel RealSense D435 RGBD camera, all wired to ROS 2 via OmniGraph
action graphs.

## ✅ Done

| # | Task | File |
|---|------|------|
| 1 | Research all references (NVIDIA forum, GitHub discussion #685, ctu-mrs CSV, OmniPerception/IsaacLab docs) | — |
| 2 | Pulled the 800k-row `mid360-real-centr.csv` from `ctu-mrs/Mid360_simulation_plugin` to `/tmp/mid360_test.csv` | `/tmp/mid360_test.csv` |
| 3 | Wrote CSV→JSON converter (`build_mid360_assets.py`) — splits 800k rows into 4 beam groups × 40 emitter states × 1 250 emitters = 200 000 emitters total | `build_mid360_assets.py` |
| 4 | Fixed the elevation convention bug (CSV is **zenith-from-down**; RTX `elevationDeg` is **from horizontal** → `elevation = 90 − zenith`); after fix: −7.21° to +52.16° matches the published Mid360 spec | `build_mid360_assets.py` |
| 5 | Generated `mid360_emitter_states.json` (10.5 MB, valid for 4 beam groups × 40 states) | `mid360_emitter_states.json` |
| 6 | Discovered the G1 USD's link structure (`/g1_29dof_with_hand_rev_1_0/torso_link/{mid360_link,d435_link,head_link,logo_link}`) | (verified via pxr) |
| 7 | Confirmed the G1 USD is at `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/G1/g1.usd` (HTTP 200) | — |
| 8 | Used the modern API per `isaacsim-sensor` skill: `Lidar("/path", attributes={...})` from `isaacsim.sensors.experimental.rtx` (not legacy `omni.isaac.sensor`) | `build_mid360_unitree_g1.py` |
| 9 | Used `ROS2Publish*` / `ROS2SubscribeJointState` OmniGraph nodes from `isaacsim.ros2.bridge` per `isaac-sim-ros2-bridge` skill (not legacy `LidarHelper`) | `build_mid360_unitree_g1.py` |
| 10 | Used the headless `SimulationApp({"headless": True, "renderer": "RayTracedLighting"})` bootstrap per `isaac-sim-headless-deployment` skill | `build_mid360_unitree_g1.py` |
| 11 | Wrote main build script with: G1 reference, mount-link finder, 4 LiDAR prims, D435 camera, ground plane, action graphs for /clock, /tf, /joint_states, /joint_efforts, /joint_command (subscriber), /livox/mid360/{a,b,c,d}, /d435/{rgb,depth,points} | `build_mid360_unitree_g1.py` |

## ⚠️ Known issues hit while testing (in run order)

1. **`SimulationApp` not bootstrapping** — needed EULA pre-acceptance (you handled this manually).
2. **`UsdGeom.SetStageMetersPerUnit(1.0)`** — must pass the stage as the first arg (it's a static USD-method, not a stage-method).
3. **Ground plane**: `omni.kit.commands.execute("CreatePrim", attrs={...})` failed (`attrs` not a valid kwarg). Switched to `isaacsim.core.experimental.objects.GroundPlane("/World/ground_plane")`.
4. **LiDAR attribute schema**: many attributes from the reference Gist (`scanType`, `scanRateBaseHz`, `tickRate`, `stateResolutionStep`, `numberOfChannels`, `numberOfEmitters`, `numLines`, `numRaysPerLine`, `elementsCoordsType`, `outputFrameOfReference`, `rayType`, `rotationDirection`, `pulseTimeNs`, etc.) **do not exist on the `OmniSensorGenericLidarCoreAPI` schema in Isaac Sim 6.0.1**. The `Lidar()` constructor's `attributes={}` dict rejects unknown attributes with `ValueError: No attribute 'omni:sensor:Core:tickRate' exists`.

   Authoritative valid Core attributes (from `generatedSchema.usda`):
   ```
   aspectRatio, azimuthErrorMean, azimuthErrorStd, beamWaistHorM, beamWaistVerM,
   bitDepthResolution, calibrationGain, divergenceHorDeg, divergenceVerDeg,
   effectiveApertureSizeM, elevationErrorMean, elevationErrorStd, farRangeM,
   focusDistM, intensityScalePercent, maxAzimuthROI, minAzimuthROI,
   minDistBetweenEchosM, minReflectance, minReflectionRangeM, Msquared,
   nearRangeM, peakPowerW, pixelPitch, quantumEfficiency, rangeAccuracyM,
   rangeOffsetM, rangeResolutionM, reflectionPowerFraction,
   startAzimuthOffsetDeg, transmissionPowerFraction, validEndAzimuthDeg,
   validStartAzimuthDeg, waveLengthNm
   bools: accumulateOutputs, instantLidar, skipDroppingInvalidPoints
   ```
   `tickRate` lives at `omni:sensor:tickRate` (no `:Core:`), set via the
   `Lidar(..., tick_rate=...)` constructor parameter, not via `attributes={}`.

## ⏭ Remaining work (in dependency order)

### Step 1 — Trim `lidar_attributes()` to schema-valid attributes only

In `build_mid360_unitree_g1.py`, the `lidar_attributes()` function currently has ~30
invalid attributes. Verified the actual schema in
`omni.usd.schema.omni_sensors-0.0.0+69cbf6ad/usd_plugins/generatedSchema.usda` —
only ONE `bool omni:sensor:Core:*` attribute exists in 6.0.1: `skipDroppingInvalidPoints`.
The PROGRESS.md note about `accumulateOutputs` and `instantLidar` was based on a stale
5.x-era list — they do **not** exist in 6.0.1 and must be dropped.

Minimum safe attribute set (verified against the 6.0.1 generatedSchema):
```python
return {
    "omni:sensor:Core:nearRangeM": LIDAR_NEAR_M,
    "omni:sensor:Core:farRangeM": LIDAR_FAR_M,
    "omni:sensor:Core:waveLengthNm": LIDAR_WAVELENGTH_NM,
    "omni:sensor:Core:azimuthErrorMean": 0.0,
    "omni:sensor:Core:azimuthErrorStd": 0.05,
    "omni:sensor:Core:elevationErrorMean": 0.0,
    "omni:sensor:Core:elevationErrorStd": 0.05,
    "omni:sensor:Core:skipDroppingInvalidPoints": True,
}
```
Pass `tick_rate=LIDAR_TICK_HZ` to the `Lidar(...)` constructor (lives at
`omni:sensor:tickRate`, not `omni:sensor:Core:tickRate`).

### Step 1b — Fix emitter-state attribute types to UIntArray

Verified against the schema: `fireTimeNs` and `channelId` are `uint[]` (not `int[]`).
`azimuthDeg` and `elevationDeg` are `float[]`. Update `create_lidar_prim()`:
```python
a = prim.CreateAttribute(f"omni:sensor:Core:emitterState:{key}:azimuthDeg",
                          Sdf.ValueTypeNames.FloatArray)
a.Set(state["azimuthDeg"])
a = prim.CreateAttribute(f"omni:sensor:Core:emitterState:{key}:elevationDeg",
                          Sdf.ValueTypeNames.FloatArray)
a.Set(state["elevationDeg"])
a = prim.CreateAttribute(f"omni:sensor:Core:emitterState:{key}:fireTimeNs",
                          Sdf.ValueTypeNames.UIntArray)
a.Set(np.array(state["fireTimeNs"], dtype=np.uint32).tolist())
a = prim.CreateAttribute(f"omni:sensor:Core:emitterState:{key}:channelId",
                          Sdf.ValueTypeNames.UIntArray)
a.Set(np.array(state["channelId"], dtype=np.uint32).tolist())
```

### Step 1c — Export BOTH `.usd` (binary) and `.usda` (text)

User wants both formats so the asset can be opened in Isaac Sim GUI (fast binary
load) AND inspected/edited by humans or AI agents (human-readable USDA).
Update `main()` after the existing export:
```python
info(f"Exporting USD → {OUTPUT_USD}")
stage.GetRootLayer().Export(str(OUTPUT_USD))
info(f"Saved ({OUTPUT_USD.stat().st_size / 1024:.1f} KiB)")

OUTPUT_USDA = OUTPUT_USD.with_suffix(".usda")
info(f"Exporting USDA → {OUTPUT_USDA}")
stage.GetRootLayer().Export(str(OUTPUT_USDA))
info(f"Saved ({OUTPUT_USDA.stat().st_size / 1024:.1f} KiB)")
```
The `.usda` file extension triggers the USD text serializer automatically — no
extra args needed. Files will be ~5-10× larger than binary but still well under
the 50 MB cap.

### Step 2 — Verify `Lidar()` accepts the trimmed attribute set

Re-run `python build_mid360_unitree_g1.py`. Expect to see all 4 LiDAR prims
created successfully. If `Lidar()` still rejects an attribute, drop it — the
schema defaults are sane.

### Step 3 — Verify emitter-state attributes

After the script runs, re-open the saved USD and inspect one of the
`/World/G1/torso_link/mid360_link/Livox_Mid360_*/omni:sensor:Core:emitterState:s001:azimuthDeg`
attributes. Expected: a 1250-element float array, range roughly [−180, +180].

### Step 4 — Verify action-graph nodes exist

The OmniGraph nodes I used (e.g. `isaacsim.ros2.bridge.ROS2PublishPointCloud`)
should exist in 6.0.1 but some may be deprecated. If a `make_graph()` call
fails with "type not registered", check the Isaac Sim 6.0 docs for the
replacement node name. Likely candidates:
- `isaacsim.ros2.bridge.ROS2PublishPointCloud` ✓
- `isaacsim.ros2.bridge.ROS2PublishJointState` ✓
- `isaacsim.ros2.bridge.ROS2PublishEffortState` ← verify this exists; the
  skill only listed `PublishJointState`, not `PublishEffortState`. If it
  doesn't exist, drop the `/joint_efforts` topic and document it as a TODO.
- `isaacsim.ros2.bridge.ROS2SubscribeJointState` ✓
- `isaacsim.ros2.bridge.ROS2PublishImage` ← needs verification.
- `isaacsim.core.nodes.IsaacApplyJointCommands` ← verify exact node name and
  attribute names (`inputs:articulationPrim`, `inputs:commandType`,
  `inputs:jointCommands`).

### Step 5 — D435 depth publish

Currently I publish `/d435/depth_image` (single-channel depth image) via
`ROS2PublishImage`. For depth as a PointCloud2, we need a depth annotator
attached to a Replicator render product. The proper pattern is:
1. Create a render product on `/World/D435/RGB`.
2. Attach `distance_to_image_plane` + `distance_to_camera` annotators.
3. Wire those to `ROS2PublishPointCloud` with `type: "depth_pcl"`.

For Isaac Sim 6.0.1, the right path is `rep.create.render_product(path)`
followed by `rep.annotators.get("distance_to_image_plane").attach([rp])`,
then connect to the OG node. **Will require a rework of the depth graph
build** — currently the simplest path is just the RGB Image + depth Image,
and skip the depth-PCL until the annotator pipeline is working.

### Step 6 — Save the USD **and** the USDA

After the script runs, BOTH files must exist:
- `mid360_unitree_g1.usd` — binary, ~5 MB, fast to load in Isaac Sim
- `mid360_unitree_g1.usda` — text, ~25–50 MB, human-readable for editing/debugging

Both use the same `Export()` call — only the file extension differs. USD's
serializer picks text vs binary format from the extension automatically.

To verify they exist and are non-trivial in size:
```bash
ls -lh /home/thakk100/Projects/thesis/mid360_unitree_g1.{usd,usda}
head -50 /home/thakk100/Projects/thesis/mid360_unitree_g1.usda   # confirm readable text
```

### Step 7 — `README_mid360_g1.md`

Cover: how to open the USD, the topic list, how to re-run the build, the
schema attribute caveat (some attributes from the 5.1.0 reference don't exist
in 6.0.1).

### Step 8 — End-to-end verification

1. Re-run `python build_mid360_unitree_g1.py` (headless). Check exit code = 0.
2. Inspect the saved USD: `python -c "from pxr import Usd; s=Usd.Stage.Open('/home/thakk100/Projects/thesis/mid360_unitree_g1.usd'); print([p.GetPath() for p in s.Traverse() if 'Livox' in p.GetName()])"` → expect 4 prims.
3. Open the USD in Isaac Sim 6.0.1 GUI, press Play. In an RVIZ2 instance (in the same ROS_DOMAIN_ID), expect:
   - `/clock`, `/tf`, `/tf_static` flowing
   - `/joint_states` flowing (29 DOF + hands)
   - 4 PointCloud2 topics on `/livox/mid360/{a,b,c,d}` at ~10 Hz each
   - `/d435/rgb`, `/d435/depth_image` flowing

## Risk register

| Risk | Mitigation |
|------|------------|
| `ROS2PublishEffortState` / `ROS2PublishImage` node not registered in 6.0.1 | Drop the topic from the script and document; verify via `og.get_registry().get_node_types()` |
| Some `omni:sensor:Core:*` attributes from the reference Gist were renamed in 6.0 | Trim to schema-valid set, accept defaults for the rest |
| The PROGRESS.md's earlier "authoritative" Core-attr list included `accumulateOutputs` / `instantLidar` — **neither exists in the 6.0.1 schema** (verified: only `skipDroppingInvalidPoints` is defined as `bool omni:sensor:Core:*`) | Use only the 8-attribute minimum set in Step 1a; rely on schema defaults for the rest |
| Stage export exceeds 5 MB per prim for the serialized emitter-state arrays | Reference user showed 25 states × 5 000 emitters = 1.9 MB; we have 40 × 1 250 ≈ 0.3 MB per prim × 4 prims = 1.2 MB total — comfortably below the cap |
| USDA text export is ~5–10× larger than binary (~25–50 MB total) | Acceptable trade-off for human readability; if too large we can drop the `.usda` and regenerate from `.usd` via `usdcat` later |
| The `OnPlaybackTick` is on `/World/ROS2Bridge/...` graphs but the LiDAR publishes inside `/World/G1/torso_link/mid360_link/Livox_Mid360_X` — verify the `sourcePrim` paths in the publish nodes resolve correctly across graph boundaries | Set `sourcePrim` to the absolute LiDAR path; OG nodes typically resolve relative to the graph parent, so absolute paths are safest |
| The G1 USD references to `configuration/...` USDs (from S3) won't resolve when the saved USD is reopened on a different machine without internet | Acceptable trade-off for now; the README should note "requires internet on first open to resolve G1 variants" |

## Files

| Path | Status |
|------|--------|
| `/home/thakk100/Projects/thesis/build_mid360_assets.py` | ✅ working (CSV → JSON) |
| `/home/thakk100/Projects/thesis/mid360_emitter_states.json` | ✅ generated (10.5 MB) |
| `/home/thakk100/Projects/thesis/build_mid360_unitree_g1.py` | ⚠️ in progress (attr schema mismatch on lidar_attributes()) |
| `/home/thakk100/Projects/thesis/mid360_unitree_g1.usd` | ❌ not yet produced (binary) |
| `/home/thakk100/Projects/thesis/mid360_unitree_g1.usda` | ❌ not yet produced (human-readable text) |
| `/home/thakk100/Projects/thesis/README_mid360_g1.md` | ❌ not yet written |

## Next session checklist

1. ⬜ Trim `lidar_attributes()` to schema-valid attributes (Step 1a). Drop
   `accumulateOutputs` / `instantLidar` (don't exist in 6.0.1 schema).
2. ⬜ Fix emitter-state attribute types to UIntArray for fireTimeNs/channelId (Step 1b).
3. ⬜ Add `OUTPUT_USDA = OUTPUT_USD.with_suffix('.usda')` and call `Export()` twice (Step 1c).
4. ⬜ Run script, expect 4 LiDAR prims created (Step 2).
5. ⬜ Verify emitter-state arrays survive both exports (Step 3).
6. ⬜ Verify each action-graph node type exists (Step 4).
7. ⬜ Simplify D435 publish to RGB + depth image only (skip depth-PCL) (Step 5).
8. ⬜ Re-run, expect both .usd AND .usda produced (Step 6).
9. ⬜ Write README (Step 7).
10. ⬜ Open in GUI + RVIZ2 to verify (Step 8).