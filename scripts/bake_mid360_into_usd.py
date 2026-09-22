"""Bake the Livox Mid-360 RTX LiDAR into portable USD assets.

Today the OmniLidar prim exists only after ``g1_sim.rtx_lidar.spawn_mid360``
has read ``assets/lidar_configs_rotary/Livox_Mid360_R.json`` and authored the
scan pattern onto ``/World/G1/mid360_link`` at runtime - open
``assets/g1_29dof_sensors.usd`` in any other Isaac Sim/Lab project and the
robot has no lidar. The JSON is consulted ONLY at authoring time: everything
(emitter states, firing/report rates, ranges, CARTESIAN coords, tick rate)
is plain USD attributes on the prim. This script performs that authoring
ONCE against the asset and exports:

    assets/g1_29dof_sensors_mid360.usd      layer-stack copy: original robot
                                             USD (configuration/ references
                                             intact) + OmniLidar authored
                                             into its root layer
    assets/g1_29dof_sensors_mid360_flat.usd  stage.Flatten() of the above: ONE
                                             self-contained binary USD - copy
                                             it (plus nothing) into another
                                             project
    assets/mid360_omnilidar.usda            standalone sensor subtree, ASCII,
                                             referenceable under ANY link of
                                             ANY robot (identity transform)

No custom class is needed to consume any of these:

  * schema  - built-in ``OmniLidar`` (omni.usd.schema.omni_sensors), applied
              by string name. Target needs Isaac Sim 5+/6 (4.x lacks the
              schema); Isaac Lab 3.0 runs on Isaac Sim 6, so fine.
  * runtime - built-in ``isaacsim.sensors.experimental.rtx.LidarSensor``
              wrapper (+ ``parse_generic_model_output_data``), exactly what
              ``g1_sim/rtx_publisher.py`` uses. Isaac Lab 3.0 itself ships
              NO RTX-lidar class - its only lidar is ``ray_caster``'s
              ``LidarPatternCfg`` (CPU ray casts against collision meshes,
              different physics entirely) - so keep reading the OmniLidar
              through the Isaac Sim wrapper from Isaac Lab code too.
  * ROS2    - optional and consumer-side (``g1_sim.rtx_publisher``); not part
              of the asset, nothing ROS2 is baked in.

Requires one Kit boot; the bootstrap (charset pin, extension enable) is the
proven path shared with ``scripts/capture_screenshots.py`` (import
``g1_warehouse_sim`` with headless/no-ros2 argv).

USAGE
    cd ~/Projects/thesis/G1_sim && set -a && source .envrc && set +a
    python scripts/bake_mid360_into_usd.py             # bake + static verify
    python scripts/bake_mid360_into_usd.py --fire-test # + boot the flat export
                                                       #   and confirm points emit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROBOT_USD = REPO / "assets/g1_29dof_sensors.usd"
CONFIG_DIR = REPO / "assets/lidar_configs_rotary"
STACK_USD = REPO / "assets/g1_29dof_sensors_mid360.usd"
FLAT_USD = REPO / "assets/g1_29dof_sensors_mid360_flat.usd"
STANDALONE_USDA = REPO / "assets/mid360_omnilidar.usda"

# Proven Kit boot: g1_warehouse_sim's module import creates SimulationApp
# (charset pin + RTX LiDAR launch settings live there); its parser gets
# headless/no-ros2 so nothing warehouse- or ROS-related runs.
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--fire-test", action="store_true",
                     help="after baking, open the flat export and confirm the "
                          "lidar emits points through the built-in LidarSensor "
                          "wrapper (one extra ~60 s)")
_cli = _parser.parse_args()

sys.argv = ["g1_warehouse_sim.py", "--headless", "--no-ira", "--no-ros2"]
import g1_warehouse_sim as gw  # noqa: E402  (boots SimulationApp)

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from g1_sim.rtx_lidar import MID360_QUAT_WXYZ, spawn_mid360  # noqa: E402

# Exactly the args scripts/g1_warehouse_sim.py passes in the proven run.
SIM_TRANSLATION = (0.0, 0.0, -0.05)
IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


def log(msg: str) -> None:
    print(f"[BAKE] {msg}", flush=True)


def _quat_close(q: Gf.Quatd, wxyz: tuple[float, float, float, float], tol: float = 1e-6) -> bool:
    w, x, y, z = wxyz
    return (
        abs(q.GetReal() - w) < tol
        and all(abs(a - b) < tol for a, b in zip(q.GetImaginary(), (x, y, z)))
    )


def find_mid360_link(stage: Usd.Stage) -> str:
    direct = "/g1_29dof/mid360_link"
    if stage.GetPrimAtPath(direct).IsValid():
        return direct
    for prim in stage.Traverse():
        if prim.GetName() == "mid360_link":
            return str(prim.GetPath())
    raise SystemExit("[BAKE] mid360_link not found in robot USD")


def verify(path: Path, profile: dict, expect_translation, expect_quat) -> bool:
    """Reopen an export and prove the lidar survived the round trip.

    Checks the things that would silently break consumption elsewhere:
    prim type, applied API schemas (string names - what a foreign Isaac Sim
    uses to instantiate the schema), key scalar attributes, s001 emitter-state
    arrays against the source JSON, and the local Xform.
    """
    ok = True

    def check(cond: bool, label: str) -> None:
        nonlocal ok
        if not cond:
            ok = False
        log(f"    {'PASS' if cond else 'FAIL'}  {label}")

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        log(f"    FAIL  reopen {path}")
        return False
    lidars = [p for p in stage.Traverse() if p.GetTypeName() == "OmniLidar"]
    check(len(lidars) == 1, f"exactly one OmniLidar prim (found {len(lidars)})")
    if not lidars:
        return False
    prim = lidars[0]
    log(f"    prim: {prim.GetPath()}")

    apis = prim.GetAppliedSchemas()
    check("OmniSensorGenericLidarCoreAPI" in apis, "OmniSensorGenericLidarCoreAPI applied")
    n_states = len(profile["emitterStates"])
    expected = {f"OmniSensorGenericLidarCoreEmitterStateAPI:s{i + 1:03d}" for i in range(n_states)}
    missing = expected - set(apis)
    check(not missing, f"{n_states} EmitterStateAPI:s001.. instances applied"
                       + (f" (missing {sorted(missing)[:3]}...)" if missing else ""))

    def attr(name: str):
        a = prim.GetAttribute(name)
        return a.Get() if a and a.HasValue() else None

    check(attr("omni:sensor:Core:elementsCoordsType") == "CARTESIAN",
          "elementsCoordsType = CARTESIAN")
    check(attr("omni:sensor:tickRate") == 10.0, "tickRate = 10.0")
    if "patternFiringRateHz" in profile:
        check(attr("omni:sensor:Core:reportRateBaseHz") == int(profile["patternFiringRateHz"]),
              f"reportRateBaseHz = {int(profile['patternFiringRateHz'])}")
    if "numLines" in profile:
        check(attr("omni:sensor:Core:numLines") == int(profile["numLines"]),
              f"numLines = {profile['numLines']}")

    # Emitter state s001: full round trip against the source JSON (USD stores
    # float32, hence the tolerance).
    state0 = profile["emitterStates"][0]
    az = attr("omni:sensor:Core:emitterState:s001:azimuthDeg")
    el = attr("omni:sensor:Core:emitterState:s001:elevationDeg")
    az_ok = az is not None and len(az) == len(state0["azimuthDeg"]) and all(
        abs(float(a) - float(b)) <= 2e-3 for a, b in zip(az, state0["azimuthDeg"]))
    el_ok = el is not None and len(el) == len(state0["elevationDeg"]) and all(
        abs(float(a) - float(b)) <= 2e-3 for a, b in zip(el, state0["elevationDeg"]))
    check(az_ok, f"s001 azimuthDeg matches JSON ({len(state0['azimuthDeg'])} values)")
    check(el_ok, f"s001 elevationDeg matches JSON ({len(state0['elevationDeg'])} values)")

    # Local transform of the sensor prim itself. GetLocalTransformation()
    # returns a bare Gf.Matrix4d on this USD build (a (matrix, bool) tuple on
    # others) - handle both.
    xf = UsdGeom.Xformable(prim)
    got = xf.GetLocalTransformation()
    mat = got[0] if isinstance(got, tuple) else got
    tr = mat.ExtractTranslation()
    tr_ok = all(abs(a - b) < 1e-5 for a, b in zip(tuple(tr), expect_translation))
    q = mat.ExtractRotationQuat()
    q_ok = _quat_close(q, expect_quat)
    check(tr_ok, f"local translation {tuple(round(v, 4) for v in tr)}")
    check(q_ok, "local orientation matches expected wxyz quaternion")

    return ok


def mesh_stats(stage: Usd.Stage) -> tuple[int, int]:
    """Count meshes with geometry anywhere in the stage.

    The G1's visuals are instanced: a default Traverse skips instance
    prototypes (where the meshes actually live) and reports zero - so scan
    both the normal prim range and every instance prototype. Works whether or
    not Flatten() kept the instances.
    """
    n_mesh, n_pts = 0, 0

    def scan(root_prim=None) -> None:
        nonlocal n_mesh, n_pts
        it = Usd.PrimRange.Stage(stage) if root_prim is None else Usd.PrimRange(root_prim)
        for prim in it:
            if prim.IsA(UsdGeom.Mesh):
                pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if pts:
                    n_mesh += 1
                    n_pts += len(pts)

    scan()
    for proto in stage.GetPrototypes():
        scan(proto)
    return n_mesh, n_pts


def fire_test() -> bool:
    """Open the flat export, drop in test geometry, read through the built-in
    LidarSensor wrapper - the same consumption path another project would use."""
    from isaacsim.sensors.experimental.rtx import (
        LidarSensor,
        parse_generic_model_output_data,
    )

    log(f"fire test: opening {FLAT_USD.name}")
    ctx = omni.usd.get_context()
    ctx.open_stage(str(FLAT_USD))
    stage = ctx.get_stage()
    if stage is None:
        log("    FAIL  could not open flat export")
        return False
    # Let Kit finish processing the new stage before authoring into it.
    # Authoring during open_stage's background layer churn throws
    # pxr.Tf.ErrorException: "Detected usd threading violation ... Concurrent
    # changes to layer(s)" (seen on both Scene.Define and SimulationContext's
    # auto-create); pumping a few frames settles serials (same pattern as
    # capture_screenshots.py after build_scene_fallback).
    for _ in range(10):
        omni.kit.app.get_app().update()

    # RTX ray-traces rendered geometry: floor + 4 walls around the robot.
    ground = UsdGeom.Cube.Define(stage, "/BakeTest/Ground")
    ground.AddScaleOp().Set(Gf.Vec3f(60.0, 60.0, 0.2))   # cube size=2 -> 120x120x0.4
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.2))  # top face at z=0
    for i, tr in enumerate([(12, 0, 6), (-12, 0, 6), (0, 12, 6), (0, -12, 6)]):
        wall = UsdGeom.Cube.Define(stage, f"/BakeTest/Wall{i}")
        wall.AddScaleOp().Set(Gf.Vec3f(0.2, 12.0, 6.0))
        wall.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in tr]))

    lidar_path = str([p for p in stage.Traverse() if p.GetTypeName() == "OmniLidar"][0].GetPath())
    log(f"    wrapping {lidar_path} with LidarSensor")
    sensor = LidarSensor(lidar_path, annotators=["generic-model-output"])

    # SimulationContext, not SimulationApp: .step(render=True) lives on
    # isaacsim.core.api.SimulationContext (SimulationApp has no step() on
    # 6.1 - confirmed by AttributeError, and matches capture_screenshots.py).
    # The flat stage ships no PhysicsScene, and SimulationContext's auto-create
    # raised Tf.ErrorException on this build - pre-define one to skip that path.
    from isaacsim.core.api import SimulationContext
    from pxr import UsdPhysics

    if not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
        UsdPhysics.Scene.Define(stage, "/BakeTest/PhysicsScene")
        log("    defined /BakeTest/PhysicsScene (stage had none)")
    for _ in range(5):  # settle our own cube/scene authoring before sim init
        omni.kit.app.get_app().update()
    sim_ctx = SimulationContext()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    total = 0
    frames_with_points = 0
    for i in range(240):
        sim_ctx.step(render=True)
        data, _info = sensor.get_data("generic-model-output")
        if data is None:
            continue
        gmo = parse_generic_model_output_data(data)
        if gmo.x is None or len(gmo.x) == 0:
            continue
        frames_with_points += 1
        total += len(gmo.x)
        if i % 60 == 0:
            log(f"    frame {i:>3}  points={len(gmo.x)}  running total={total}")
    timeline.stop()

    ok = total >= 1000
    log(f"    {'PASS' if ok else 'FAIL'}  {total} points across {frames_with_points}/240 frames")
    return ok


def main() -> int:
    gw.enable_extensions()

    profiles = sorted(CONFIG_DIR.glob("Livox_Mid360_*.json"))
    if not profiles:
        log(f"FAIL: no Livox_Mid360_*.json in {CONFIG_DIR}")
        return 1
    if len(profiles) != 1:
        log(f"WARN: {len(profiles)} profiles in rotary dir; verifying against first")
    profile = json.loads(profiles[0].read_text())["profile"]
    log(f"source profile: {profiles[0].name} ({len(profile['emitterStates'])} emitter states)")

    ctx = omni.usd.get_context()
    all_ok = True

    # --- 1. standalone sensor subtree (identity transform, any robot) --------
    log(f"authoring standalone subtree -> {STANDALONE_USDA.name}")
    ctx.new_stage()
    spawn_mid360(
        "",
        config_dir=CONFIG_DIR,
        translation=(0.0, 0.0, 0.0),
        orientation=IDENTITY_QUAT_WXYZ,
    )
    ctx.get_stage().GetRootLayer().Export(str(STANDALONE_USDA))

    # --- 2. robot USD + sensor (proven sim args) ----------------------------
    log(f"authoring lidar into {ROBOT_USD.name} -> {STACK_USD.name}")
    ctx.open_stage(str(ROBOT_USD))
    stage = ctx.get_stage()
    if stage is None:
        log("FAIL: could not open robot USD")
        return 1
    mount = find_mid360_link(stage)
    log(f"mount: {mount}")
    prim_paths = spawn_mid360(
        mount,
        config_dir=CONFIG_DIR,
        translation=SIM_TRANSLATION,
        orientation=MID360_QUAT_WXYZ,
    )
    log(f"spawned: {prim_paths}")
    stage.GetRootLayer().Export(str(STACK_USD))

    # --- 3. flatten to a single self-contained file -------------------------
    log(f"flattening composed stage -> {FLAT_USD.name} (references/payloads inlined)")
    flat_layer = stage.Flatten()
    flat_layer.Export(str(FLAT_USD))

    for p in (STACK_USD, FLAT_USD, STANDALONE_USDA):
        log(f"  {p.name}: {p.stat().st_size / (1024 * 1024):.2f} MiB")

    flat_stage = Usd.Stage.Open(str(FLAT_USD))
    n_mesh, n_pts = mesh_stats(flat_stage)
    deps = [str(d) for d in flat_stage.GetRootLayer().GetCompositionAssetDependencies() if str(d)]
    log(f"flat file geometry: {n_mesh} mesh prims, {n_pts} points; "
        f"external asset refs: {len(deps)}"
        + ("" if not deps else f" -> {deps[:3]}{' ...' if len(deps) > 3 else ''}"))
    # Release both stage handles BEFORE fire_test re-opens flat.usd through
    # Kit's UsdContext. Two UsdStage objects observing the SAME root layer made
    # authoring throw pxr.Tf.ErrorException "Detected usd threading violation
    # ... Concurrent changes to layer(s)" (serial mismatch) - fire_test's
    # Scene.Define/SimulationContext never got past it. `stage` (robot USD) is
    # stale by then too (Kit logs "still resident" warnings when switching).
    del stage, flat_stage

    # --- 4. static verification --------------------------------------------
    log("verifying layer-stack export:")
    all_ok &= verify(STACK_USD, profile, SIM_TRANSLATION, MID360_QUAT_WXYZ)
    log("verifying flat export:")
    all_ok &= verify(FLAT_USD, profile, SIM_TRANSLATION, MID360_QUAT_WXYZ)
    log("verifying standalone subtree:")
    all_ok &= verify(STANDALONE_USDA, profile, (0.0, 0.0, 0.0), IDENTITY_QUAT_WXYZ)

    # --- 5. optional fire test ---------------------------------------------
    if _cli.fire_test:
        log("fire test:")
        all_ok &= fire_test()

    if all_ok:
        log("DONE - all checks passed")
        log("copy assets/g1_29dof_sensors_mid360_flat.usd into another project; "
            "read it with isaacsim.sensors.experimental.rtx.LidarSensor - no JSON, no custom class")
        return 0
    log("DONE - with FAILURES (see above)")
    return 1


if __name__ == "__main__":
    # Kit installs its own excepthook and once reported exit code 0 after a
    # real crash - trace explicitly and judge results by the [BAKE] DONE line,
    # not the exit code. No explicit simulation_app.close(): Kit
    # auto-shuts-down on interpreter exit, and closing mid-LidarSensor teardown
    # segfaulted once (observed SIGSEGV).
    try:
        _code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
        _code = 1
    raise SystemExit(_code)
