#!/usr/bin/env python3
"""Headless check that every Dex3 fingertip contact sensor reports contact.

Loads the Dex3 G1 (gravity off, so it hangs in place) on a ground plane,
places a small kinematic cube overlapping each fingertip's distal link, steps
physics and asserts that all 6 tip sensors report ``in_contact`` and a
non-zero force from their raw contact impulses (the value the ROS publisher
sends). Nothing touches the tips in normal standing, and the asset disables
self-collision, so this is the direct test of the sensing path.

    python scripts/test_dex3_contacts.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Pin a coherent charset_normalizer before Kit boots (see g1_warehouse_sim.py).
import charset_normalizer.api  # noqa: F401
import charset_normalizer.md  # noqa: F401
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

manager = omni.kit.app.get_app().get_extension_manager()
for ext in ("isaacsim.core.api", "isaacsim.core.prims", "isaacsim.core.utils"):
    manager.set_extension_enabled_immediate(ext, True)

from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.sensors.experimental.physics import ContactSensor  # noqa: E402

from g1_sim.dex3_contacts import TIP_LINKS, net_force  # noqa: E402
from g1_sim.g1_robot import load_g1  # noqa: E402

ROBOT = "/World/G1"

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.Scene.Define(stage, "/World/physicsScene").CreateGravityMagnitudeAttr(9.81)
ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.CreateSizeAttr(1.0)
UsdGeom.Xformable(ground).AddScaleOp().Set(Gf.Vec3f(10.0, 10.0, 0.02))
UsdGeom.Xformable(ground).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

g1 = load_g1(
    prim_path=ROBOT, translation=(0.0, 0.0, 1.0),
    camera=False, lidar=False, imu_pelvis=False, imu_torso=False, imu_lidar=False, imu_camera=False,
    ros2=False, robot_state=False, create_articulation=False,
)
assert len(g1.tip_contacts) == 6, f"expected 6 tip sensors, got {g1.tip_contacts}"

for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)

# Probe at the centre of each tip's collision geometry (the link origin sits
# at the joint, and the mirrored right hand extends the other way).
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy])
for tip, link in TIP_LINKS.items():
    pos = bbox.ComputeWorldBound(stage.GetPrimAtPath(f"{ROBOT}/{link}/collisions")).ComputeAlignedRange().GetMidpoint()
    cube = UsdGeom.Cube.Define(stage, f"/World/probe_{tip}")
    cube.CreateSizeAttr(0.015)
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr(True)

sim = SimulationContext(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
sim.reset()
sensors = {tip: ContactSensor(path) for tip, path in g1.tip_contacts.items()}

peak = {tip: 0.0 for tip in sensors}
touched = {tip: False for tip in sensors}
for _ in range(90):
    sim.step(render=False)
    for tip, sensor in sensors.items():
        peak[tip] = max(peak[tip], float(np.linalg.norm(net_force(sensor))))
        touched[tip] |= bool(sensor.get_data()["in_contact"])

ok = True
for tip in sensors:
    good = touched[tip] and peak[tip] > 0.0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {tip:14s} in_contact={touched[tip]}  peak |F| = {peak[tip]:.2f} N")
print("RESULT:", "PASS" if ok else "FAIL", flush=True)
# SimulationApp.close() ends the process with status 0, so exit first.
os._exit(0 if ok else 1)
