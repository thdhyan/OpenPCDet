# Initializing the RTX LiDAR (Livox Mid-360)

Authoritative init sequence for the G1's Mid-360 RTX LiDAR on **Isaac Sim 6.0+**.
This is the path the warehouse sim (`scripts/g1_warehouse_sim.py`) actually
uses. It replaces the earlier warp / IsaacLab `ray_caster` LiDAR, which has been
removed — RTX ray-traces the real rendered scene, so self-occlusion and dynamic
objects come for free.

References: Isaac Sim 6.0.1 RTX LiDAR docs
(`isaacsim_sensors_rtx_lidar.html`) and the emitter-state discussion
[isaac-sim/IsaacSim#685](https://github.com/isaac-sim/IsaacSim/discussions/685).

---

## The five steps

### 1. Launch with LiDAR buffering on the CPU and motion BVH enabled

```python
SimulationApp({
    "headless": ...,
    # CPU-side LiDAR return buffer — avoids CUDA race (discussion #685:
    # GPU still writing while we read → "GMO magic number is not correct").
    "/app/sensors/nv/lidar/outputBufferOnGPU": False,
    # Motion BVH required for RTX LiDAR to fire. Without it Isaac Sim logs
    # "Multi-tick is enabled but motion BVH is not active" and returns 0 points.
    # SimulationApp translates this key to:
    #   --/renderer/raytracingMotion/enabled=true
    #   --/renderer/raytracingMotion/enableHydraEngineMasking=true
    # Must be set at launch — carb.settings after startup has no effect.
    "enable_motion_bvh": True,
})
```

### 2. Spawn the sensor **as** the `mid360_link` prim, with an identity transform

`g1_sim.rtx_lidar.spawn_mid360(parent_prim_path, translation, orientation)`
authors the `OmniLidar` prims directly through `pxr.Usd` (not through
`isaacsim.sensors.experimental.rtx.Lidar(attributes=...)`, whose replicator
helper star-unpacks large arrays and crashes). It applies
`OmniSensorGenericLidarCoreAPI` + one `...EmitterStateAPI:sNNN` per state.

**Mount it on the USD's existing `mid360_link` prim with an identity local
transform:**

```python
mount = f"{ROBOT_PRIM}/mid360_link"          # NOT torso_link
spawn_mid360(mount, translation=(0, 0, 0), orientation=(1, 0, 0, 0))
```

Why this exact mount is the whole ball game — see [Frames](#frames-the-below-ground-bug).

### 3. Per-prim sensor attributes (set inside `spawn_mid360`)

- `omni:sensor:tickRate = 10.0` — on 6.0 multi-tick rendering is on by default,
  so `tickRate` genuinely limits the render rate. (On 5.1 it is ignored: every
  RTX sensor renders at the sim frame rate, firing ~6x too often in a 60 Hz sim
  and corrupting the point rate — one reason 6.0+ is required.)
- `omni:sensor:Core:accumulateOutputs = True`.
- `numLines` / `numRaysPerLine` — required, or the RTX engine rejects every
  param update with `bankId 0 ... greater than the profile numLines 0`.
- The non-repetitive sweep is expressed as `emitterState:sNNN:{azimuthDeg,
  elevationDeg,fireTimeNs,channelId,...}` arrays, one state per frame, generated
  from the real `mid360.npy` scan pattern (see [Regenerating](#regenerating-the-profiles)).
  Hydra caps one prim at ~5 MB of emitter data, so the pattern is split across
  several co-located prims whose union is the full pattern.

### 4. Read returns through the `LidarSensor` runtime wrapper — not a raw annotator

Per the 6.0.1 docs: *raw prim creation alone is insufficient; you must use the
`LidarSensor` runtime wrapper to enable annotators.* Manually attaching
`rep.AnnotatorRegistry.get_annotator(...)` to a hand-made render product returns
a valid-looking annotator whose `"data"` array is **always empty**.

```python
from isaacsim.sensors.experimental.rtx import LidarSensor, parse_generic_model_output_data
sensor = LidarSensor(prim_path, annotators=["generic-model-output"])
data, _info = sensor.get_data("generic-model-output")
gmo = parse_generic_model_output_data(data)
x, y, z = gmo.x, gmo.y, gmo.z          # Cartesian point arrays, sensor frame
```

`g1_sim.rtx_publisher.RtxLidarPublisher` wraps this: it merges every prim,
accumulates thin per-frame slices into a full sweep, filters junk (non-finite,
exact-origin no-hit points, and anything past the profile's `farRangeM`, which
can only be buffer-corruption garbage), and publishes `sensor_msgs/PointCloud2`.

### 5. Publish over rclpy — **not** the OmniGraph `ROS2RtxLidarHelper`

On this Isaac Sim build the menu-generated `isaacsim.ros2.bridge.ROS2RtxLidarHelper`
graph **advertises** `/livox/mid360/points` but never emits on it, while still
spending 4 render products competing for GPU memory. `RtxLidarPublisher`
(plain rclpy) is what actually publishes. Drive it from the sim loop:

```python
publisher = RtxLidarPublisher(prim_paths, topic=[...], publish_rate=10.0)
while running:
    sim.step(render=True)
    publisher.accumulate()             # every step: grab this frame's slice
    publisher.publish(sim.current_time)  # emits at publish_rate
```

---

## Frames: the below-ground bug

The single most important rule: **the ROS `frame_id` the cloud is published in
must be the same prim the points originate from.**

`GenericModelOutput.frameOfReference` is `SENSOR` for this sensor, so `x/y/z`
come out in the **sensor prim's own frame**. `PublishTransformTree` (in
`rtx_camera.attach_robot_state_publishers`) emits a TF for every robot link,
including `mid360_link`. If you publish the points as `frame_id="mid360_link"`
but the sensor prim is actually a *different* prim (e.g. mounted on `torso_link`
with a hand-authored 180° roll quaternion), then:

- the points already carry the sensor prim's roll, **and**
- RViz applies the `mid360_link` TF's roll a **second** time,

flipping every `+elevation` return to `-elevation` and dropping the whole cloud
below the ground. That was the "points below ground" bug (fixed 2026-08-11).

The fix removes the ambiguity entirely: mount the sensor **as** `mid360_link`
(identity local transform). The URDF pose baked into that prim is then the one
and only transform, and it is exactly the TF RViz uses. `RtxLidarPublisher` also
reads `frameOfReference` at startup and relabels the cloud to `World` if the
engine ever hands back world-frame points, so a convention change can't silently
resurrect the bug.

`mid360_link` survives conversion because `convert_g1_urdf_to_usd.py` runs the
importer with `merge_fixed_joints=False`; with merging on, the importer collapses
the sensor links into the torso and the prim path vanishes.

---

## Regenerating the profiles

The emitter-state JSON in `assets/lidar_configs*/` is generated from the real
Mid-360 scan pattern:

```bash
python scripts/gen_mid360_rtx_config.py   # reads assets/scan_patterns/mid360.npy
```

Both the pattern and the G1 URDF/meshes are now vendored under `assets/`
(`assets/scan_patterns/mid360.npy`, `assets/robot/g1_29/`); there is no longer
any dependency on the external OmniPerception checkout.

---

## Version / gotcha checklist

| Symptom | Cause | Fix |
|---|---|---|
| Points form a thin +elevation band, no floor returns | `elementsCoordsType=SPHERICAL` (default) — `gmo.x/y/z` are az(deg)/el(deg)/range(m), not Cartesian; degree values ≈ small positive metres → upper-hemisphere artifact | `spawn_mid360` sets `omni:sensor:Core:elementsCoordsType="CARTESIAN"`; publisher has SPHERICAL→Cartesian fallback |
| Points below ground | frame_id ≠ sensor prim (double roll) | mount **as** `mid360_link`, identity transform |
| Empty `"data"` array | raw annotator attach | use `LidarSensor` runtime wrapper |
| Topic advertised, silent | OG `ROS2RtxLidarHelper` on this build | publish via rclpy (`RtxLidarPublisher`) |
| `bankId 0 > numLines 0` | `numLines`/`numRaysPerLine` unset | author them in `spawn_mid360` |
| 0 points, "motion BVH not active" warning | `enable_motion_bvh` not set at launch | add `"enable_motion_bvh": True` to `SimulationApp({...})` — must be at launch, carb.settings post-startup ignored |
| Points at 95–165 m vs `farRangeM=40` | CUDA buffer race (#685) | `outputBufferOnGPU=False` + `max_range_m` filter |
| Sensor fires 6× too fast | Isaac Sim 5.1 ignores `tickRate` | use Isaac Sim 6.0+ |
