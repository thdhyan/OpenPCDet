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
    python scripts/bake_mid360_into_usd.py --fire-test # FIRST point-read test of
                                                       #   the existing exports in a
                                                       #   pristine session (stage swaps
                                                       #   kill the sensor engine), then
                                                       #   bake fresh ones; run twice to
                                                       #   validate a new bake
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
                     help="before baking, reference the EXISTING exports into "
                          "the pristine session stage and confirm they emit "
                          "points via the built-in LidarSensor wrapper "
                          "(one extra ~60 s; bake once first to create them)")
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
    """Prove the exports emit points when consumed the way another project
    would: REFERENCE them into the pristine boot stage, wrap with the built-in
    LidarSensor wrapper, read back returns.

    Never ctx.open_stage() first: every open_stage run returned 0 points/0
    frames from BOTH a baked and a live spawn_mid360 prim in the same session
    (time advancing normally, attr/schema diff NONE between them) - and those
    sessions had already gone through main's new_stage()+open_stage(robot)
    as well. capture_screenshots/g1_warehouse NEVER replace the boot stage
    and the sensor works there, so pristine-session + reference is the proven
    shape (and referencing into a project is the documented consumption
    pattern anyway).

    Three prims A/B'd in one session:
      live - spawn_mid360 authored fresh (capture's proven baseline)
      flat - reference into g1_29dof_sensors_mid360_flat.usd at its authored
             prim path (the single-file copy-into-another-project export)
      usda - reference into mid360_omnilidar.usda (identity standalone)
    Main() calls this BEFORE any stage swap; run a bake once first so the
    exports exist.
    """
    from isaacsim.sensors.experimental.rtx import (
        LidarSensor,
        parse_generic_model_output_data,
    )

    log("fire test: consuming exports by reference into the pristine session stage")
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()  # boot stage - never replaced (capture parity)
    if stage is None:
        log("    FAIL  no session stage")
        return False
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # RTX ray-traces rendered geometry: floor + 4 walls around the robot.
    ground = UsdGeom.Cube.Define(stage, "/BakeTest/Ground")
    ground.AddScaleOp().Set(Gf.Vec3f(60.0, 60.0, 0.2))   # cube size=2 -> 120x120x0.4
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.2))  # top face at z=0
    for i, tr in enumerate([(12, 0, 6), (-12, 0, 6), (0, 12, 6), (0, -12, 6)]):
        wall = UsdGeom.Cube.Define(stage, f"/BakeTest/Wall{i}")
        wall.AddScaleOp().Set(Gf.Vec3f(0.2, 12.0, 6.0))
        wall.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in tr]))

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

    # Wrap AFTER SimulationContext exists - the proven order in both working
    # scripts (capture_screenshots.py: sim at line 117, LidarSensor at 218;
    # g1_warehouse_sim.py likewise).
    stage.DefinePrim("/BakeTest/World", "Xform")
    live_path = spawn_mid360(
        "/BakeTest/World",
        config_dir=CONFIG_DIR,
        translation=(0.0, 0.0, 2.0),
        orientation=MID360_QUAT_WXYZ,
    )[0]

    # Consume both exports by primPath reference - what another project does.
    # USD semantics: a reference puts the TARGET prim's content AT THE
    # REFERRING PRIM's path (it does not create a child named after the
    # target) - so the referrer must itself be the typeless Livox_Mid360_R
    # prim, under a positioned Xform holder that keeps its own xform ops.
    # (Statically validated with pure pxr: composed type=OmniLidar, s001
    # azimuth 128 values, apiSchemas metadata intact.)
    stage.DefinePrim("/BakeTest/Flat", "Xform")
    flat_holder = stage.GetPrimAtPath("/BakeTest/Flat")
    UsdGeom.Xformable(flat_holder).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 2.0))
    flat_path = "/BakeTest/Flat/Livox_Mid360_R"
    flat_ref = stage.DefinePrim(flat_path)  # typeless; type composes from target
    flat_ref.GetReferences().AddReference(
        str(FLAT_USD), "/g1_29dof/mid360_link/Livox_Mid360_R"
    )

    stage.DefinePrim("/BakeTest/Usda", "Xform")
    usda_holder = stage.GetPrimAtPath("/BakeTest/Usda")
    UsdGeom.Xformable(usda_holder).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 4.0))
    usda_path = "/BakeTest/Usda/Livox_Mid360_R"
    usda_ref = stage.DefinePrim(usda_path)
    usda_ref.GetReferences().AddReference(str(STANDALONE_USDA), "/Livox_Mid360_R")

    paths = {"live": live_path, "flat": flat_path, "usda": usda_path}
    for k, p in paths.items():
        pr = stage.GetPrimAtPath(p)
        if not pr.IsValid() or pr.GetTypeName() != "OmniLidar":
            log(f"    FAIL  [{k}] {p}: valid={pr.IsValid()} type={pr.GetTypeName()!r} "
                "- reference did not compose")
            return False
    log(f"    prims: {paths}")

    def _sig(path: str) -> tuple[set, set]:
        pr = stage.GetPrimAtPath(path)
        names = {a.GetName() for a in pr.GetAttributes() if a.HasAuthoredValue()}
        return names, set(pr.GetAppliedSchemas())

    f_names, f_sch = _sig(flat_path)
    l_names, l_sch = _sig(live_path)
    log(f"    attr-diff flat-only : {sorted(f_names - l_names) or 'NONE'}")
    log(f"    attr-diff live-only : {sorted(l_names - f_names) or 'NONE'}")
    log(f"    schema-diff flat-only: {sorted(f_sch - l_sch) or 'NONE'}  "
        f"live-only: {sorted(l_sch - f_sch) or 'NONE'}")

    # Capture parity: an explicit render product per sensor prim before the
    # wrap (capture_screenshots.py:198). No writer: its list-attach API is
    # unimplemented on this build ("Attaching a list of render products is
    # currently not implemented") and it is debug-viz only.
    import omni.replicator.core as rep  # noqa: E402

    for p in paths.values():
        rep.create.render_product(p, [1, 1])

    sensors = {k: LidarSensor(p, annotators=["generic-model-output"]) for k, p in paths.items()}
    log(f"    wrapped {list(sensors)}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    def _now() -> float:
        """Sim time for diagnostics - API name differs across builds."""
        for obj, attr in ((sim_ctx, "current_time"), (timeline, "get_current_time")):
            try:
                v = getattr(obj, attr)
                return float(v() if callable(v) else v)
            except Exception:
                continue
        return -1.0

    for _ in range(10):  # render-product warmup before the first read
        sim_ctx.step(render=True)
    totals = {k: 0 for k in sensors}
    frames_pts = {k: 0 for k in sensors}
    n_none = {k: 0 for k in sensors}
    n_empty = {k: 0 for k in sensors}
    for i in range(240):
        sim_ctx.step(render=True)
        for k, sensor in sensors.items():
            data, _info = sensor.get_data("generic-model-output")
            if data is None:
                n_none[k] += 1
                continue
            gmo = parse_generic_model_output_data(data)
            if gmo.x is None or len(gmo.x) == 0:
                n_empty[k] += 1
                if i == 0:
                    try:
                        names = getattr(getattr(data, "dtype", None), "names", None)
                        row0 = data[0] if len(data) else None
                        nz = ""
                        if names and "x" in names:
                            import numpy as _np

                            nz = f" nonzero_x={int(_np.count_nonzero(data['x']))}"
                        log(f"    [{k}] frame0 empty: dtype_names={names} "
                            f"row0={row0!r}{nz} t={_now():.3f}")
                    except Exception as exc:  # noqa: BLE001
                        log(f"    [{k}] frame0 diag failed: {exc}")
                continue
            frames_pts[k] += 1
            totals[k] += len(gmo.x)
            if i == 0:
                log(f"    [{k}] frame0: {len(gmo.x)} pts  t={_now():.3f}")
    t_end = _now()
    timeline.stop()

    for k in sensors:
        ok_k = totals[k] >= 1000
        log(f"    {'PASS' if ok_k else 'FAIL'}  [{k}] {totals[k]} pts across "
            f"{frames_pts[k]}/240 frames (data=None {n_none[k]}, empty {n_empty[k]})")
    log(f"    sim time advanced to t={t_end:.3f}")
    # The bake's claim is that the FILES work: flat + standalone must emit.
    # (live = session-health baseline; logged above either way.)
    return totals["flat"] >= 1000 and totals["usda"] >= 1000


def main() -> int:
    gw.enable_extensions()
    # Mirror capture_screenshots.py line 99: the .nodes extension registers the
    # RTX lidar node-graph node types the sensor pipeline ticks on.
    _mgr = omni.kit.app.get_app().get_extension_manager()
    _mgr.set_extension_enabled_immediate("isaacsim.sensors.rtx.nodes", True)
    for _ in range(5):
        omni.kit.app.get_app().update()

    profiles = sorted(CONFIG_DIR.glob("Livox_Mid360_*.json"))
    if not profiles:
        log(f"FAIL: no Livox_Mid360_*.json in {CONFIG_DIR}")
        return 1
    if len(profiles) != 1:
        log(f"WARN: {len(profiles)} profiles in rotary dir; verifying against first")
    profile = json.loads(profiles[0].read_text())["profile"]
    log(f"source profile: {profiles[0].name} ({len(profile['emitterStates'])} emitter states)")

    # --- 0. optional fire test: the ONLY thing --fire-test runs -------------
    # Two constraints collide in one process: (a) the RTX sensor engine dies
    # with stage swaps - every open_stage/new_stage-after-unwrap run returned
    # 0/240 frames from live AND baked prims, while capture/warehouse (boot
    # stage never replaced) emit fine - so fire must run FIRST on a pristine
    # session; (b) the bake's steps 1-2 swap stages, which SEGFAULTS with
    # the fire test's live sensors still attached (observed exit 139 after
    # the fire PASS). So --fire-test = fire only; bake in a separate
    # invocation, then fire-test the result.
    if _cli.fire_test:
        if not (FLAT_USD.exists() and STACK_USD.exists() and STANDALONE_USDA.exists()):
            log("FAIL: --fire-test needs existing exports (run one bake first)")
            return 1
        log("fire test (pre-existing exports, pristine session):")
        return 0 if fire_test() else 1

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

    # --- 5. (fire test is a separate invocation: --fire-test) ----------------

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
