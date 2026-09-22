#!/usr/bin/env python3
"""Standalone Mid-360 LiDAR visualizer inside Isaac Sim (no ROS2, no robot).

Spawns a flat Mesh ground plane + UsdPhysics.Scene, places the Mid-360 at a
configurable height, runs the sim, and prints per-prim Cartesian diagnostics
every PRINT_EVERY steps. In GUI mode also draws coloured debug points in the
viewport (blue=low elevation, red=high).

ROS2 compatibility: this script uses the same spawn_mid360 + LidarSensor path
as g1_warehouse_sim.py / RtxLidarPublisher. Adding ROS2 publishing here is a
one-liner (wrap sensors in RtxLidarPublisher instead of reading manually).

Usage (from G1_sim/ with the uv venv active):

    python scripts/lidar_viz_isaacsim.py --headless --num-prims 2 --frames 60
    python scripts/lidar_viz_isaacsim.py            --num-prims 2 --frames 60
    python scripts/lidar_viz_isaacsim.py --headless --num-prims 4 --frames 120

Isolation checklist:
  - Flat Mesh ground (RTX ray-traces Mesh, not Cube/Plane implicit prims).
  - UsdPhysics.Scene present (suppresses motion BVH warning, enables RTX firing).
  - 10-update pre-warm before timeline.play() (avoids GMO magic-number errors).
  - 30-step post-play warmup window skipped before collecting (buffer stabilise).
  - elementsCoordsType=CARTESIAN confirmed at runtime from GMO.

If points appear here but not in the full sim → self-occlusion or robot mount.
If no points here → prim attributes or LidarSensor wiring wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--headless", action="store_true", help="No viewport - print stats only.")
parser.add_argument("--height", type=float, default=0.5, help="Sensor height above ground (m).")
parser.add_argument("--num-prims", type=int, default=4, help="Number of Mid-360 prims to spawn.")
parser.add_argument("--frames", type=int, default=0, help="Stop after N sim steps (0=run forever).")
parser.add_argument(
    "--config-dir",
    type=str,
    default="assets/lidar_configs",
    help="Directory with Livox_Mid360_*.json profiles.",
)
parser.add_argument(
    "--enclose",
    action="store_true",
    help="Add 4 walls around the ground so rays in every direction hit "
    "geometry - lets the printed az/el stats measure the FIRING pattern "
    "instead of the scene's silhouette.",
)
args = parser.parse_args()

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        # CPU-side LiDAR return buffer — avoids CUDA race (discussion #685).
        "/app/sensors/nv/lidar/outputBufferOnGPU": False,
        # Disable multi-tick via extra_args (command-line path — the only way
        # that takes effect before the RTX sensor extension loads and checks it).
        # SimulationApp config dict keys are carb settings set AFTER extensions
        # load; /rtx/hydra/supportMultiTickRate is read at extension init time,
        # so the config-dict path is too late. extra_args passes it as a kit
        # command-line flag, which is set before any extension starts.
        # Disabling multi-tick removes the "multi-tick + no motion BVH" block
        # without needing enable_motion_bvh (which OOMs 8GB VRAM at shader compile).
        "extra_args": ["--/rtx/hydra/supportMultiTickRate=false"],
    }
)

# ── imports after SimulationApp ──────────────────────────────────────────────
import math

import carb
import numpy as np
import omni.timeline
import omni.usd
from isaacsim.sensors.experimental.rtx import LidarSensor, parse_generic_model_output_data
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from g1_sim.rtx_lidar import MID360_QUAT_WXYZ, spawn_mid360

# ── build a minimal scene ────────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# World Xform
world = stage.DefinePrim("/World", "Xform")

# Physics scene — required for motion BVH (multi-tick RTX LiDAR warning).
# Without this Isaac Sim logs "motion BVH not active" and may refuse to fire.
physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
physics_scene.CreateGravityMagnitudeAttr(9.81)

# Flat Mesh ground — RTX LiDAR ray-traces Mesh prims, not Plane/Cube.
# Cube/Plane are implicit shapes the RTX renderer does not ray-trace.
mesh_prim = UsdGeom.Mesh.Define(stage, "/World/ground")
hw = 20.0  # half-width metres
mesh_prim.CreatePointsAttr([Gf.Vec3f(-hw,-hw,0), Gf.Vec3f(hw,-hw,0),
                             Gf.Vec3f(hw,hw,0),   Gf.Vec3f(-hw,hw,0)])
mesh_prim.CreateFaceVertexCountsAttr([4])
mesh_prim.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
mesh_prim.CreateNormalsAttr([Gf.Vec3f(0,0,1)]*4)
UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())

if args.enclose:
    # 4 vertical walls (8 m tall) at the ground's edge so rays in every
    # azimuth/elevation direction return a hit. Without this only the
    # down-going fraction of the pattern is observable.
    wall_h = 8.0
    for i, (cx, cy, w, d) in enumerate([
        (0.0, -hw, 2 * hw, 0.1),
        (0.0,  hw, 2 * hw, 0.1),
        (-hw, 0.0, 0.1, 2 * hw),
        ( hw, 0.0, 0.1, 2 * hw),
    ]):
        wall = UsdGeom.Mesh.Define(stage, f"/World/wall_{i}")
        zc = wall_h / 2.0
        hx, hy = w / 2.0, d / 2.0
        wall.CreatePointsAttr([
            Gf.Vec3f(cx-hx, cy-hy, 0), Gf.Vec3f(cx+hx, cy-hy, 0),
            Gf.Vec3f(cx+hx, cy+hy, 0), Gf.Vec3f(cx-hx, cy+hy, 0),
            Gf.Vec3f(cx-hx, cy-hy, zc*2), Gf.Vec3f(cx+hx, cy-hy, zc*2),
            Gf.Vec3f(cx+hx, cy+hy, zc*2), Gf.Vec3f(cx-hx, cy+hy, zc*2),
        ])
        wall.CreateFaceVertexCountsAttr([4] * 6)
        wall.CreateFaceVertexIndicesAttr([0,1,2,3, 4,5,6,7, 0,1,5,4, 1,2,6,5, 2,3,7,6, 3,0,4,7])
        UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
    print(f"[VIZ] enclosure walls added (hw={hw}, h={wall_h})")

# Sensor mount Xform at height above ground.
mount_path = "/World/sensor_mount"
mount = stage.DefinePrim(mount_path, "Xform")
xf = UsdGeom.Xformable(mount)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, args.height))

# Spawn Mid-360 prims on the mount.
# 180° roll matches URDF mid360_joint rpy=(3.14, 0, 0).
prim_paths = spawn_mid360(
    mount_path,
    config_dir=REPO / args.config_dir,
    translation=(0.0, 0.0, 0.0),
    orientation=MID360_QUAT_WXYZ,
)
if args.num_prims and args.num_prims < len(prim_paths):
    prim_paths = prim_paths[: args.num_prims]

print(f"[VIZ] spawned {len(prim_paths)} LiDAR prim(s) at height={args.height:.2f}m")
for p in prim_paths:
    print(f"      {p}")

# Pump one update before timeline.play() so the render pipeline (Hydra,
# RTX sensor backend) initialises with the stage already populated.
# Without this the first several GMO reads hit an uninitialised buffer
# and log "magic number is not correct".
print("[VIZ] pre-warming render pipeline (10 updates before play)...")
for _ in range(10):
    simulation_app.update()

# Wrap each prim in a LidarSensor runtime object (required to enable annotators).
sensors = [LidarSensor(p, annotators=["generic-model-output"]) for p in prim_paths]

# ── optional viewport debug-draw ─────────────────────────────────────────────
_draw = None
if not args.headless:
    try:
        from omni.debugdraw import get_debug_draw_interface
        _draw = get_debug_draw_interface()
        print("[VIZ] debug-draw available — points will render in viewport")
    except Exception:
        print("[VIZ] debug-draw unavailable — headless stats only")

def _el_to_color(el_deg: float) -> int:
    """Map elevation angle to ABGR uint32 for debug draw: blue(low)→red(high)."""
    t = max(0.0, min(1.0, (el_deg + 7.2) / 59.4))  # -7.2..+52.2 → 0..1
    r = int(t * 255)
    b = int((1 - t) * 255)
    return (0xFF << 24) | (b << 16) | (0 << 8) | r  # ABGR

def _gather_cartesian(sensor, prim_idx: int):
    """Return (N,3) Cartesian array for one prim, or None."""
    data, _ = sensor.get_data("generic-model-output")
    if data is None:
        return None
    gmo = parse_generic_model_output_data(data)
    if gmo.x is None or len(gmo.x) == 0 or gmo.numElements == 0:
        return None

    coords = getattr(gmo, "elementsCoordsType", None)
    coords_name = getattr(coords, "name", str(coords)) if coords is not None else ""

    if "SPHERICAL" in coords_name.upper():
        az = np.radians(np.asarray(gmo.x, dtype=np.float32))
        el = np.radians(np.asarray(gmo.y, dtype=np.float32))
        r  = np.asarray(gmo.z, dtype=np.float32)
        rxy = r * np.cos(el)
        xyz = np.stack([rxy * np.cos(az), rxy * np.sin(az), r * np.sin(el)], axis=1)
    else:
        xyz = np.stack([
            np.asarray(gmo.x, dtype=np.float32),
            np.asarray(gmo.y, dtype=np.float32),
            np.asarray(gmo.z, dtype=np.float32),
        ], axis=1)

    # Filter: finite, non-origin, within 40 m.
    r2 = np.linalg.norm(xyz, axis=1)
    keep = np.isfinite(xyz).all(axis=1) & (r2 > 1e-4) & (r2 <= 40.0)
    xyz = xyz[keep]

    # Log coords type once.
    if not hasattr(_gather_cartesian, "_logged"):
        _gather_cartesian._logged = set()
    if prim_idx not in _gather_cartesian._logged:
        print(f"[VIZ] prim {prim_idx}: elementsCoordsType={coords_name}  numElements={gmo.numElements}")
        _gather_cartesian._logged.add(prim_idx)

    return xyz if len(xyz) > 0 else None

# ── simulation loop ──────────────────────────────────────────────────────────
timeline = omni.timeline.get_timeline_interface()
timeline.play()

step = 0
# Skip the first WARMUP_STEPS frames while RTX buffer stabilises.
# GMO "magic number" errors appear during this window — normal, not a bug.
# At scanRateBaseHz=10 and 60 Hz sim, one full scan = 6 steps.
# 30 steps = 5 full scans: enough for the RTX pipeline to reach steady state.
WARMUP_STEPS = 30
PRINT_EVERY = 30  # steps between diagnostics after warmup
point_counts = [0] * len(prim_paths)
el_ranges = [None] * len(prim_paths)
az_ranges = [None] * len(prim_paths)
az_hists = [np.zeros(12, dtype=np.int64) for _ in prim_paths]

try:
    while True:
        simulation_app.update()
        step += 1

        if step <= WARMUP_STEPS:
            if step == WARMUP_STEPS:
                print(f"[VIZ] warmup done ({WARMUP_STEPS} steps), starting collection")
            continue

        for i, sensor in enumerate(sensors):
            xyz = _gather_cartesian(sensor, i)
            if xyz is None:
                continue
            point_counts[i] += len(xyz)

            # Track elevation range from Cartesian (sensor-local frame).
            r = np.linalg.norm(xyz, axis=1)
            nonzero = r > 1e-4
            if np.any(nonzero):
                el = np.degrees(np.arcsin(np.clip(xyz[nonzero, 2] / r[nonzero], -1, 1)))
                lo, hi = float(el.min()), float(el.max())
                el_ranges[i] = (lo, hi) if el_ranges[i] is None else (
                    min(el_ranges[i][0], lo), max(el_ranges[i][1], hi)
                )
                az = np.degrees(np.arctan2(xyz[nonzero, 1], xyz[nonzero, 0]))
                alo, ahi = float(az.min()), float(az.max())
                az_ranges[i] = (alo, ahi) if az_ranges[i] is None else (
                    min(az_ranges[i][0], alo), max(az_ranges[i][1], ahi)
                )
                az_hists[i] += np.histogram(az, bins=12, range=(-180, 180))[0]

            # Draw points in viewport (sub-sample to keep frame time reasonable).
            if _draw is not None and len(xyz) > 0:
                stride = max(1, len(xyz) // 500)
                pts_draw = xyz[::stride]
                for pt in pts_draw:
                    el_approx = math.degrees(math.asin(float(np.clip(pt[2] / (np.linalg.norm(pt) + 1e-9), -1, 1))))
                    color = _el_to_color(el_approx)
                    _draw.draw_point(
                        carb.Float3(float(pt[0]), float(pt[1]), float(pt[2] + args.height)),
                        color,
                        3.0,
                    )

        collect_step = step - WARMUP_STEPS
        if collect_step % PRINT_EVERY == 0:
            total = sum(point_counts)
            print(f"\n[VIZ] collect_step={collect_step}  total_pts={total}")
            for i in range(len(prim_paths)):
                er = f"el=[{el_ranges[i][0]:.1f},{el_ranges[i][1]:.1f}]" if el_ranges[i] else "el=no_returns"
                ar = f"az=[{az_ranges[i][0]:.1f},{az_ranges[i][1]:.1f}]" if az_ranges[i] else "az=no_returns"
                hist = " ".join(f"{v:5d}" for v in az_hists[i])
                print(f"  prim {i}: {point_counts[i]} pts  {ar}  {er}")
                print(f"           az hist(-180..180, 30deg): {hist}")
            point_counts = [0] * len(prim_paths)
            el_ranges = [None] * len(prim_paths)
            az_ranges = [None] * len(prim_paths)
            az_hists = [np.zeros(12, dtype=np.int64) for _ in prim_paths]

        if args.frames > 0 and step >= (args.frames + WARMUP_STEPS):
            print(f"[VIZ] reached --frames {args.frames} post-warmup, stopping.")
            break

except KeyboardInterrupt:
    print("[VIZ] interrupted.")
finally:
    timeline.stop()
    simulation_app.close()
