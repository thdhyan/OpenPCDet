#!/usr/bin/env python3
"""Inspect and validate Mid-360 OmniLidar prim attributes without running the full sim.

Spawns a minimal Isaac Sim stage, calls spawn_mid360, then reads back every
authored attribute and checks them against expected values. Exits 0 on pass,
1 on any failure.

Usage (uv venv active, from G1_sim/):

    python scripts/test_lidar_prim.py
    python scripts/test_lidar_prim.py --headless   # always use headless for CI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--config-dir", type=str, default="assets/lidar_configs")
args = parser.parse_args()

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

simulation_app = SimulationApp({"headless": True, "/app/sensors/nv/lidar/outputBufferOnGPU": False})

import omni.usd
from pxr import Gf, UsdGeom
from g1_sim.rtx_lidar import MID360_QUAT_WXYZ, spawn_mid360

# ── minimal stage ─────────────────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
mount = stage.DefinePrim("/World/mount", "Xform")

print("=" * 60)
print("test_lidar_prim: spawning Mid-360 prims")
print("=" * 60)

prim_paths = spawn_mid360(
    "/World/mount",
    config_dir=REPO / args.config_dir,
    translation=(0.0, 0.0, 0.0),
    orientation=MID360_QUAT_WXYZ,
)
print(f"spawned {len(prim_paths)} prim(s): {prim_paths}\n")

REQUIRED_ATTRS = {
    "omni:sensor:tickRate": 10.0,
    "omni:sensor:Core:accumulateOutputs": True,
    "omni:sensor:Core:elementsCoordsType": "CARTESIAN",
    "omni:sensor:Core:scanRateBaseHz": 10,
    "omni:sensor:Core:numLines": 1,
}

FAILURES = []

for path in prim_paths:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        FAILURES.append(f"{path}: prim not found on stage")
        continue
    if prim.GetTypeName() != "OmniLidar":
        FAILURES.append(f"{path}: type={prim.GetTypeName()!r}, expected 'OmniLidar'")

    print(f"── {path}")

    for attr_name, expected in REQUIRED_ATTRS.items():
        attr = prim.GetAttribute(attr_name)
        if not attr.IsValid():
            FAILURES.append(f"  {path}: attr '{attr_name}' missing")
            print(f"  FAIL  {attr_name}: MISSING (expected {expected!r})")
            continue
        val = attr.Get()
        # Coerce for comparison.
        if isinstance(expected, float):
            match = abs(float(val) - expected) < 1e-4
        elif isinstance(expected, int):
            match = int(val) == expected
        else:
            match = str(val) == str(expected)
        status = "PASS" if match else "FAIL"
        if not match:
            FAILURES.append(f"  {path}: '{attr_name}' = {val!r}, expected {expected!r}")
        print(f"  {status}  {attr_name} = {val!r}  (expected {expected!r})")

    # Check emitter state schemas applied.
    schemas = prim.GetAppliedSchemas()
    emitter_schemas = [s for s in schemas if "EmitterStateAPI" in s]
    print(f"  INFO  applied schemas: {len(emitter_schemas)} EmitterStateAPI instances")
    if not emitter_schemas:
        FAILURES.append(f"  {path}: no EmitterStateAPI schemas applied")

    # Check state s000 has azimuth/elevation arrays spanning full Mid-360 range.
    s0_az = prim.GetAttribute("omni:sensor:Core:emitterState:s000:azimuthDeg")
    s0_el = prim.GetAttribute("omni:sensor:Core:emitterState:s000:elevationDeg")
    if s0_az.IsValid() and s0_el.IsValid():
        import numpy as np
        az = np.array(s0_az.Get())
        el = np.array(s0_el.Get())
        az_ok = az.min() < -90 and az.max() > 90
        el_ok = el.min() < 0 and el.max() > 10
        print(f"  {'PASS' if az_ok else 'FAIL'}  s000 azimuth=[{az.min():.1f}, {az.max():.1f}]  (need <-90 and >90)")
        print(f"  {'PASS' if el_ok else 'FAIL'}  s000 elevation=[{el.min():.1f}, {el.max():.1f}]  (need <0 and >10)")
        if not az_ok:
            FAILURES.append(f"{path}: s000 azimuth range [{az.min():.1f}, {az.max():.1f}] too narrow")
        if not el_ok:
            FAILURES.append(f"{path}: s000 elevation range [{el.min():.1f}, {el.max():.1f}] lacks negative values")
    else:
        FAILURES.append(f"{path}: emitterState:s000 az/el arrays missing")

    # Check local transform has the 180° roll.
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    orient_ops = [op for op in ops if "orient" in op.GetOpName().lower()]
    if orient_ops:
        q = orient_ops[0].Get()  # Gf.Quatd or Gf.Quatf
        w, x, y, z = q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]
        # 180° roll about X: w≈0, x≈1, y≈0, z≈0
        roll180 = abs(abs(x) - 1.0) < 0.01 and abs(w) < 0.01
        status = "PASS" if roll180 else "FAIL"
        if not roll180:
            FAILURES.append(f"{path}: orientation wxyz=({w:.3f},{x:.3f},{y:.3f},{z:.3f}), expected 180° roll (x≈1)")
        print(f"  {status}  orientation wxyz=({w:.3f},{x:.3f},{y:.3f},{z:.3f})  (expected 180° roll: w≈0, x≈±1)")
    else:
        FAILURES.append(f"{path}: no orient xformOp found")
        print(f"  FAIL  no orient xformOp found")

    print()

print("=" * 60)
if FAILURES:
    print(f"RESULT: FAIL — {len(FAILURES)} failure(s):")
    for f in FAILURES:
        print(f"  {f}")
    simulation_app.close()
    sys.exit(1)
else:
    print(f"RESULT: PASS — all {len(prim_paths)} prim(s) validated")
    simulation_app.close()
    sys.exit(0)
