#!/usr/bin/env python3
"""G1 in a populated warehouse: RTX LiDAR + camera + IMU + decoupled_wbc
locomotion, plus IRA-driven wandering humans and Nova Carters, on Isaac Sim 6.1.0.

    source /generalSSD/IsaacLab/isaac6/.envrc   # uv venv; or cd into the repo with direnv
    python scripts/g1_warehouse_sim.py --headless --steps 200

Extends ``g1_rtx_sim.py``'s proven sensor/locomotion pipeline (kept
untouched and still the thing to run for a bare-warehouse-free scene) with:

- a real warehouse environment, via ``g1_sim.warehouse`` (fixes a Nucleus
  path-resolution bug found while building this: ``g1_rtx_sim.py``'s Nucleus
  room loader passed a catalog-relative path straight into
  ``add_reference_to_stage``, which silently failed and fell back to flat
  ground on every run - see that module's docstring for the fix)
- wandering humans and Nova Carters spawned via IRA
  (``g1_sim.ira_actors`` - see its docstring for a hard-coded-frame-budget
  bug found and worked around there)
- robot + full sensor stack via ``g1_sim.g1_robot.load_g1`` (D435 RGB-D,
  Mid-360 LiDAR, four IMUs - pelvis/torso/Mid-360/RealSense - plus TF,
  joint states, clock; every sensor ON by default, kwargs disable each)

Scene bootstrap order matters: IRA's ``setup_simulation()`` opens the
warehouse **as a new root stage** (a full replace, not a reference), so it
must run before anything else touches the stage. If IRA is disabled
(``--no-ira``) or fails, falls back to a plain warehouse reference (still
using the fixed loader) with no dynamic actors - the pipeline degrades
gracefully rather than aborting.
"""

import argparse
import sys
import traceback
from pathlib import Path

# --- charset_normalizer forensics + defensive pin (Plan.md, run #6) --------
# Kit's ext manager re-imports charset_normalizer mid-boot from its pip
# prebundles; a mismatched cd/md pair aborts cython module init
# ("MessDetectorPlugin size changed ... Expected 24 ... got 16") and cascades
# into isaacsim.core.api / sensors.experimental.rtx import failures. Pre-load
# one coherent compiled set from site-packages *before* Kit boots, and
# audit-hook every later charset_normalizer import so the Kit log records
# which file (and which cached sys.modules state) each subsequent load used.
def _cs_audit(event, args):
    if event == "import":
        name = args[0] if args else None
        if isinstance(name, str) and name.startswith("charset_normalizer"):
            cached = {
                k: getattr(sys.modules.get(k), "__file__", "?")
                for k in sorted(sys.modules)
                if k.startswith("charset_normalizer")
            }
            sys.stderr.write(
                f"[cs-audit] import {name} file={args[1] if len(args) > 1 else None}\n"
                f"[cs-audit] cached={cached}\n"
                f"[cs-audit] sys.path[:12]={sys.path[:12]}\n"
                f"[cs-audit] stack:\n{''.join(traceback.format_stack(limit=8))}\n"
            )
            sys.stderr.flush()
    elif event == "exec":
        fname = getattr(args[0], "co_filename", "")
        if isinstance(fname, str) and "charset_normalizer" in fname:
            sys.stderr.write(f"[cs-audit] exec {fname}\n")
            sys.stderr.flush()


sys.addaudithook(_cs_audit)

import charset_normalizer
import charset_normalizer.api  # noqa: F401  - pins compiled cd/md/constants
import charset_normalizer.md

sys.stderr.write(
    f"[cs-audit] PIN pkg={charset_normalizer.__file__} "
    f"md={charset_normalizer.md.__file__} "
    f"CharInfo={charset_normalizer.md.CharInfo.__basicsize__} "
    f"MDP={charset_normalizer.md.MessDetectorPlugin.__basicsize__}\n"
)
sys.stderr.flush()
# --- end charset_normalizer forensics --------------------------------------

from isaacsim import SimulationApp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser(description="G1 in a populated warehouse.")
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--xr",
    action="store_true",
    help="Launch with isaacsim.exp.base.xr.vr kit (OpenXR/CloudXR mode). "
    "Requires 'python -m isaacteleop.cloudxr' running first as the system "
    "OpenXR compositor, and NVIDIA CloudXR client on Quest 3. "
    "Renders the scene into the VR headset; implies not headless.",
)
parser.add_argument("--steps", type=int, default=0, help="Stop after N steps; 0 runs forever.")
parser.add_argument("--no-ros2", action="store_true")
parser.add_argument("--no-camera", action="store_true", help="Skip the D435 camera (saves render time).")
parser.add_argument("--no-locomotion", action="store_true", help="Skip the decoupled_wbc cmd_vel bridge.")
parser.add_argument(
    "--wbc-mode",
    type=str,
    default="internal",
    choices=["internal", "external"],
    help="Actuation source when locomotion is enabled: 'internal' runs the "
    "in-sim decoupled_wbc ONNX policies; 'external' listens for per-joint "
    "Float64 position targets on /g1/joint/<name> (29 topics, same contract "
    "as the Gazebo wbc_node) and applies them - your own ROS node drives the "
    "robot. Gains are set in-sim in both modes.",
)
parser.add_argument("--no-ira", action="store_true", help="Skip IRA humans/carters; warehouse + G1 only.")
parser.add_argument("--num-humans", type=int, default=2, help="IRA wandering characters. 0 disables the group.")
parser.add_argument("--num-carters", type=int, default=1, help="IRA wandering Nova Carters. 0 disables the group.")
parser.add_argument("--ira-seed", type=int, default=42)
parser.add_argument(
    "--cache-scene",
    type=str,
    default="assets/warehouse_ira_baked.usd",
    help="Load IRA's post-bake scene from here if it exists, instead of "
    "re-downloading the warehouse and re-baking the navmesh every run "
    "(~60-80s). Saved automatically the first time IRA setup succeeds. "
    "Characters/carter sit static at their saved pose - not a substitute "
    "for a real run when actor wander motion matters.",
)
parser.add_argument("--rebake", action="store_true", help="Ignore --cache-scene and force a real IRA setup + navmesh bake, overwriting the cache.")
parser.add_argument(
    "--navmesh-max-frames",
    type=int,
    default=3000,
    help="Raised navmesh-bake poll budget - see g1_sim/ira_actors.py docstring for why the stock 100 fails.",
)
parser.add_argument(
    "--config-dir",
    type=str,
    default="assets/lidar_configs_rotary",
    help="Mid-360 LiDAR profile dir. Default is the ROTARY matched-envelope "
    "profile (Livox_Mid360_R, 128 emitters = 32 el ch x 4 groups, -7.2..+52.2 "
    "deg @10 Hz) - the emitter-state solidState path is dead on this Isaac "
    "Sim build (see docs/RTX_LIDAR_INIT.md). Pass assets/lidar_configs for "
    "the exact-pattern solidState configs (non-functional here).",
)
parser.add_argument("--num-prims", type=int, default=0, help="Use only the first N LiDAR sensor prims.")
parser.add_argument(
    "--keep-carter-cameras",
    action="store_true",
    help="Keep the Nova Carter Hawk/Owl camera rigs. Off by default: they are never "
    "published and their RTX passes OOM-crash an 8 GB GPU (see ira_actors.strip_carter_cameras).",
)
parser.add_argument(
    "--freeze-robot",
    action="store_true",
    help="Disable gravity on the G1 so it holds its spawn (standing) pose without a "
    "balance policy - use with --no-locomotion to test the sensors on an upright robot.",
)
args_cli = parser.parse_args()

# --xr implies rendering into the VR headset (not headless)
if args_cli.xr and args_cli.headless:
    print("[WH] WARNING: --xr and --headless are mutually exclusive; ignoring --headless")
    args_cli.headless = False

SIM_RATE_HZ = 60.0

# XR mode: use the OpenXR/CloudXR experience kit so IsaacSim renders into
# the VR compositor (CloudXR runtime from `python -m isaacteleop.cloudxr`).
# Pass the full path — SimulationApp fails to resolve the short name on some builds.
import os as _os, site as _site
_XR_KIT = _os.path.join(
    _os.path.dirname(_site.getsitepackages()[0]),  # .venv-isaac/lib/python3.12
    "site-packages/isaacsim/apps/isaacsim.exp.base.xr.vr.kit",
)
if args_cli.xr and not _os.path.exists(_XR_KIT):
    # fallback: try the short name (works on some IsaacSim builds)
    _XR_KIT = "isaacsim.exp.base.xr.vr"
# Isaac Sim 6.1.0 (Kit 110.3): Kit no longer resolves the bare experience
# name against the --ext-folder paths ("Can't read isaacsim.exp.full") —
# same failure the XR branch hit. Pass the full path when it exists.
_FULL_KIT = _os.path.join(
    _os.path.dirname(_site.getsitepackages()[0]),  # .venv/lib/python3.12
    "site-packages/isaacsim/apps/isaacsim.exp.full.kit",
)
if args_cli.xr:
    _experience = _XR_KIT
else:
    _experience = _FULL_KIT if _os.path.exists(_FULL_KIT) else "isaacsim.exp.full"

simulation_app = SimulationApp(
    {
        "headless": args_cli.headless,
        # CPU-side LiDAR return buffer — avoids CUDA race (discussion #685).
        # ALSO passed via extra_args below: several /rtx and /app/sensors
        # settings are read at extension-init time, before SimulationApp's
        # config-dict carb settings land - those dict entries silently no-op
        # (supportMultiTickRate proven no-op live 2026-08-24).
        "/app/sensors/nv/lidar/outputBufferOnGPU": False,
        # Motion BVH MUST be on when multi-tick rendering is active (6.0
        # default), else every RTX LiDAR read fails with "GMO magic number is
        # not correct" and returns nothing ("Multi-tick ... motion BVH is not
        # active. This is not supported.").
        "extra_args": [
            # EXPERIMENT (rotary bring-up 2026-08-25, kept): multi-tick ON is
            # fine for static mounts but was implicated in early bring-up
            # failures; multi-tick OFF renders sensors every sim frame without
            # the motion-BVH path. tickRate is then ignored (60 Hz firing,
            # publisher rate-limits to 10 Hz).
            "--/rtx/hydra/supportMultiTickRate=false",
            "--/renderer/raytracingMotion/enabled=true",
            "--/renderer/raytracingMotion/enableHydraEngineMasking=true",
            "--/renderer/raytracingMotion/enabledForHydraEngines=0,1,2,3",
            "--/app/sensors/nv/lidar/outputBufferOnGPU=false",
        ],
    },
    experience=_experience,
)

# Verify the sensor-critical settings ACTUALLY took effect (they are read at
# extension init; dict-only authoring has silently no-op'd before).
import carb  # noqa: E402


def _log_sensor_settings() -> None:
    s = carb.settings.get_settings()
    print(
        "[WH] sensor settings : "
        f"outputBufferOnGPU={s.get('/app/sensors/nv/lidar/outputBufferOnGPU')} "
        f"multiTickRate={s.get('/rtx/hydra/supportMultiTickRate')} "
        f"motionBVH={s.get('/renderer/raytracingMotion/enabled')}"
    )


"""Rest everything follows."""

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics, Usd

from g1_sim.arm_override import ArmTargetSubscriber
from g1_sim.g1_robot import load_g1
from g1_sim.rtx_camera import attach_cmd_vel_subscriber
from g1_sim.rtx_lidar import MID360_POS, blind_radius
from g1_sim.warehouse import WAREHOUSE_USD, build_flat_ground, load_environment

ENABLE_ROS2 = not args_cli.no_ros2
ENABLE_LOCOMOTION = not args_cli.no_locomotion
ENABLE_IRA = not args_cli.no_ira
WBC_CONTROL_HZ = 50.0  # decoupled_wbc's trained control rate

G1_USD = REPO / "assets/g1_29dof_sensors.usd"
ROBOT_PRIM = "/World/G1"
# Fallback-only static targets (used when IRA is off/unavailable, so the
# scene still has something LiDAR-visible besides the robot itself).
PEDESTRIANS = [(3.0, 0.0), (4.5, -2.0), (6.0, 2.5)]

GUI_EXTENSIONS = [
    "omni.graph.window.action",
    "omni.graph.window.generic",
    "omni.kit.widget.stage",
]


def enable_extensions() -> None:
    manager = omni.kit.app.get_app().get_extension_manager()

    # Isaac Sim 6.1.0: these live in extsDeprecated/ (deprecated since 6.0 in
    # favor of isaacsim.core.experimental.*, but still shipped) and are NOT
    # autoloaded by the experience - enable explicitly before importing.
    #   isaacsim.core.api    -> SimulationContext (main())
    #   isaacsim.core.prims  -> Articulation (main())
    #   isaacsim.core.utils  -> stage utils (g1_sim/warehouse.py)
    for ext in ("isaacsim.core.api", "isaacsim.core.prims", "isaacsim.core.utils"):
        manager.set_extension_enabled_immediate(ext, True)

    # Must precede any rclpy import - see g1_rtx_sim.py.
    manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    manager.set_extension_enabled_immediate("isaacsim.sensors.rtx", True)
    # isaacsim.sensors.physics.nodes registers the IsaacReadIMU OmniGraph node.
    # Deliberately NOT enabling the deprecated isaacsim.sensors.physics
    # extension - see spawn_imu_sensor()'s docstring for the Kit-command
    # ambiguity that causes.
    manager.set_extension_enabled_immediate("isaacsim.sensors.physics.nodes", True)

    if ENABLE_IRA:
        import g1_sim.ira_actors as ira_actors

        ira_actors.enable_extension()

    if not args_cli.headless:
        for ext in GUI_EXTENSIONS:
            manager.set_extension_enabled_immediate(ext, True)

    for _ in range(20):
        omni.kit.app.get_app().update()


def build_scene_ira() -> bool:
    """Try IRA: warehouse + humans + Nova Carters, all in one stage-owning
    call. Returns True on success."""
    import g1_sim.ira_actors as ira_actors

    config_path = REPO / "assets/ira_warehouse_config.yaml"
    ira_actors.write_config(
        config_path,
        warehouse_rel=WAREHOUSE_USD.lstrip("/"),
        num_humans=args_cli.num_humans,
        num_carters=args_cli.num_carters,
        seed=args_cli.ira_seed,
        duration_s=200.0,
    )
    print(f"[WH] IRA config      : {config_path}  ({args_cli.num_humans} humans, {args_cli.num_carters} carters)")

    return ira_actors.run_setup_blocking(
        simulation_app, config_path, max_navmesh_frames=args_cli.navmesh_max_frames
    )


def build_scene_fallback(stage) -> None:
    """Warehouse with no dynamic actors - used when --no-ira is passed or
    IRA's setup failed."""
    stage.DefinePrim("/World", "Xform")
    try:
        resolved = load_environment(stage, WAREHOUSE_USD, "/World/Env")
        print(f"[WH] environment     : {resolved}")
    except Exception as e:
        print(f"[WH] warehouse load failed ({e}), using flat ground")
        build_flat_ground(stage)

    for i, (x, y) in enumerate(PEDESTRIANS):
        box = UsdGeom.Cube.Define(stage, f"/World/targets/pedestrian_{i}")
        box.CreateSizeAttr(1.0)
        box.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.875))
        box.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 1.75))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())


def main() -> None:
    enable_extensions()
    # After enable_extensions(): Isaac Sim 6.1.0 moved isaacsim.core.api to
    # extsDeprecated — the extension must be enabled before this import.
    from isaacsim.core.api import SimulationContext

    _log_sensor_settings()

    cache_path = REPO / args_cli.cache_scene
    ira_ok = False
    if ENABLE_IRA:
        if not args_cli.rebake and cache_path.exists():
            import g1_sim.ira_actors as ira_actors

            ira_actors.load_baked_scene(cache_path)
            ira_ok = True
        else:
            ira_ok = build_scene_ira()
            if not ira_ok:
                print("[WH] IRA setup failed - falling back to warehouse with no dynamic actors")
            else:
                import g1_sim.ira_actors as ira_actors

                ira_actors.save_baked_scene(omni.usd.get_context().get_stage(), cache_path)

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    if not ira_ok:
        build_scene_fallback(stage)
    elif not args_cli.keep_carter_cameras:
        import g1_sim.ira_actors as ira_actors

        stripped = ira_actors.strip_carter_cameras(stage)
        print(f"[WH] carter cameras  : {stripped} deactivated (keep with --keep-carter-cameras)")

    # G1 always goes on top, regardless of which path built the environment.
    # load_g1 owns the reference + session-layer edit-target dance (rationale
    # in its docstring), spawns every sensor prim (each ON by default) and
    # wires the ROS2 graphs. create_articulation=False: pelvis gets wrapped
    # AFTER sim.reset() below, the order this script was verified with.
    labels = {ROBOT_PRIM: "robot"}
    if not ira_ok:
        labels.update({f"/World/targets/pedestrian_{i}": "pedestrian" for i in range(len(PEDESTRIANS))})
    g1 = load_g1(
        prim_path=ROBOT_PRIM,
        usd_path=G1_USD,
        camera=not args_cli.no_camera,
        ros2=ENABLE_ROS2,
        lidar_config_dir=REPO / args_cli.config_dir,
        lidar_num_prims=args_cli.num_prims,
        semantics=labels if (not args_cli.no_camera and ENABLE_ROS2) else None,
        create_articulation=False,
    )
    pelvis_prim = stage.GetPrimAtPath(f"{ROBOT_PRIM}/pelvis")
    print(f"[WH] G1 prim valid   : {stage.GetPrimAtPath(ROBOT_PRIM).IsValid()}  pelvis valid: {pelvis_prim.IsValid()}")

    if args_cli.freeze_robot:
        # Disable gravity on every G1 rigid body *before* sim.reset() initializes
        # the physics view, so a policy-less robot holds its standing spawn pose
        # (otherwise it collapses and the LiDAR ends up buried in the floor).
        from pxr import PhysxSchema, Usd, UsdPhysics

        frozen = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
                frozen += 1
        print(f"[WH] freeze-robot    : gravity disabled on {frozen} bodies (upright, no policy)")

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / SIM_RATE_HZ,
        rendering_dt=1.0 / SIM_RATE_HZ,
    )

    from isaacsim.core.prims import Articulation

    robot_articulation = Articulation(f"{ROBOT_PRIM}/pelvis", name="g1")
    sim.reset()
    print(f"[WH] articulation    : {robot_articulation.num_dof} DOF")
    print(f"[WH] IRA actors      : {'yes' if ira_ok else 'no'}")

    wbc_bridge = None
    joint_cmd_sub = None
    if ENABLE_LOCOMOTION:
        from g1_sim.wbc_bridge import (
            ARM_JOINTS,
            ARM_KD,
            ARM_KP,
            ALL_JOINTS,
            KD,
            KP,
            LEG_WAIST_JOINTS,
        )

        dof_names = set(robot_articulation.dof_names or [])
        missing = [j for j in ALL_JOINTS if j not in dof_names]
        if missing:
            print(f"[WH] WBC bridge      : DISABLED - USD is missing joints {missing}")
        else:
            # Trained per-joint PD gains are required in BOTH modes - the
            # uniform URDF defaults collapse the robot (see Plan.md history).
            robot_articulation.set_gains(kps=KP[None, :], kds=KD[None, :], joint_names=LEG_WAIST_JOINTS)
            robot_articulation.set_gains(
                kps=np.full((1, len(ARM_JOINTS)), ARM_KP, dtype=np.float32),
                kds=np.full((1, len(ARM_JOINTS)), ARM_KD, dtype=np.float32),
                joint_names=ARM_JOINTS,
            )
            if args_cli.wbc_mode == "external":
                import rclpy

                from g1_sim.joint_cmd_sub import JointCommandSubscriber

                # rclpy may not be initialised yet - the publisher block's
                # own init comes later, so guard here.
                if not rclpy.ok():
                    rclpy.init()
                _wbc_ros_node = rclpy.create_node("g1_joint_cmd_listener")
                joint_cmd_sub = JointCommandSubscriber(_wbc_ros_node, ALL_JOINTS)
                print(
                    "[WH] WBC mode        : EXTERNAL - waiting for /g1/joint/<name> "
                    f"Float64 targets on {len(ALL_JOINTS)} joints (robot holds default "
                    "pose until every joint receives one)"
                )
            else:
                from g1_sim.wbc_bridge import WbcBridge

                wbc_bridge = WbcBridge(
                    REPO / "assets/policy/GR00T-WholeBodyControl-Balance.onnx",
                    REPO / "assets/policy/GR00T-WholeBodyControl-Walk.onnx",
                )
                print("[WH] WBC bridge      : loaded (decoupled_wbc Balance/Walk, gains overridden)")

    mount_height = 0.8 + MID360_POS[2]
    print(f"[WH] mount height    : {mount_height:.2f} m")
    print(f"[WH] blind radius    : {blind_radius(mount_height):.2f} m")

    cmd_vel_graph_path = None
    arm_sub = None
    if ENABLE_ROS2:
        # NOT attach_ros2_publishers() here: that OG-graph path's
        # ROS2RtxLidarHelper advertises /livox/mid360/points but never
        # actually emits on this Isaac Sim build (see rtx_publisher.py's
        # docstring - load_g1()/g1_robot.py creates RtxLidarPublisher instead).
        # Calling it anyway was pure waste: 4 extra IsaacCreateRenderProduct
        # nodes/render-products for a graph whose output nothing reads,
        # competing for GPU memory with the 4 render products
        # RtxLidarPublisher itself creates, the camera's, and (new)
        # RgbdPointCloudPublisher's - live-verified 2026-08-11 as a
        # contributing cause of a mid-run crash
        # (AnnotatorRegistryError: "not attached to any render products",
        # consistent with Replicator evicting a render product under
        # memory pressure on an 8 GB GPU).
        if ira_ok:
            import g1_sim.ira_actors as ira_actors

            # Re-enabled 2026-08-10: briefly disabled on the theory that this
            # graph's "[PoseTree] eInvalid" spam was uniquely responsible for
            # a stalled boot, but the same spam (just "parent .../World"
            # eInvalid, no human targets) kept firing with this disabled -
            # so it isn't the (sole) cause and disabling it bought nothing.
            # The eInvalid warning itself is still unexplained/unfixed - see
            # FUTURE_STEPS.md - but it doesn't block this from being useful.
            actor_prims = ira_actors.discover_actor_prims(stage)
            actors_graph = ira_actors.attach_actor_tf_publishers(actor_prims)
            print(f"[WH] actors tf graph : {actors_graph}  ({len(actor_prims)} actors: {actor_prims})")

            carter_prims = ira_actors.discover_prims_at(stage, "/World/Robots/carters")
            if carter_prims:
                # Publishes IRA's own existing carter IMU (read-only) -
                # see find_imu_prim()'s docstring for why we don't create one.
                carter_imu_graphs = ira_actors.attach_carter_imu_publishers(stage, carter_prims)
                print(f"[WH] carter imus     : {len(carter_imu_graphs)}/{len(carter_prims)} found -> /carter_N/imu")

        if wbc_bridge is not None:
            cmd_vel_graph_path = attach_cmd_vel_subscriber()
            print(f"[WH] cmd_vel graph   : {cmd_vel_graph_path}  (/g1/cmd_vel)")

        import rclpy

        arm_sub = ArmTargetSubscriber(rclpy.create_node("g1_arm_override"))
        print("[WH] arm override    : /g1/arm_cmd -> ARM_JOINTS (external, hold-last)")

    else:
        print("[WH] ROS2 disabled")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print(f"[WH] timeline        : playing={timeline.is_playing()}")
    print("[WH] running\n")

    step = 0
    scans = 0
    last_points = 0
    wbc_updates = 0
    wbc_control_period = 1.0 / WBC_CONTROL_HZ
    wbc_next_time = 0.0
    try:
        while simulation_app.is_running():
            sim.step(render=True)
            step += 1

            if joint_cmd_sub is not None and sim.current_time >= wbc_next_time:
                # EXTERNAL WBC: apply whatever the outside ROS node sent.
                wbc_next_time = sim.current_time + wbc_control_period
                joint_cmd_sub.spin_once()
                targets = joint_cmd_sub.get_targets()
                if targets is not None:
                    robot_articulation.set_joint_position_targets(
                        targets[None, :].astype(np.float32), joint_names=ALL_JOINTS
                    )
                    wbc_updates += 1
                    if wbc_updates % 50 == 0:
                        print(
                            f"[WH] ext-wbc updates={wbc_updates}  "
                            f"pelvis_z={robot_articulation.get_world_poses()[0][0][2]:.3f}"
                        )

            if wbc_bridge is not None and sim.current_time >= wbc_next_time:
                from g1_sim.wbc_bridge import quat_rotate_inverse

                wbc_next_time = sim.current_time + wbc_control_period

                cmd_vx, cmd_vy, cmd_wz = 0.0, 0.0, 0.0
                if cmd_vel_graph_path is not None:
                    from g1_sim.action_graph import read_cmd_vel

                    cmd_vx, cmd_vy, cmd_wz = read_cmd_vel(graph_path=cmd_vel_graph_path)

                qpos_all = np.asarray(robot_articulation.get_joint_positions(joint_names=ALL_JOINTS))[0]
                qvel_all = np.asarray(robot_articulation.get_joint_velocities(joint_names=ALL_JOINTS))[0]
                _, quat_wxyz = robot_articulation.get_world_poses()
                quat_wxyz = np.asarray(quat_wxyz)[0]
                ang_vel_world = np.asarray(robot_articulation.get_angular_velocities())[0]
                ang_vel_body = quat_rotate_inverse(quat_wxyz, ang_vel_world)

                target = wbc_bridge.step(qpos_all, qvel_all, quat_wxyz, ang_vel_body, cmd_vx, cmd_vy, cmd_wz)
                robot_articulation.set_joint_position_targets(
                    target[None, :].astype(np.float32), joint_names=LEG_WAIST_JOINTS
                )
                if wbc_updates == 0:
                    # Shoulders stay at the default URDF zero pose (arms
                    # hanging straight down) per 2026-08-11 request - only
                    # bend the elbows (index 3 left, 10 right in ARM_JOINTS)
                    # so the forearms/palms swing forward into the D435's
                    # downward-pitched FOV instead of raising the whole arm.
                    # +1.2 bent the forearms backward instead of forward
                    # (live visual check 2026-08-11) - negated.
                    arm_pose = np.zeros((1, len(ARM_JOINTS)), dtype=np.float32)
                    arm_pose[0, 3] = -1.2    # left_elbow_joint
                    arm_pose[0, 10] = -1.2   # right_elbow_joint
                    robot_articulation.set_joint_position_targets(
                        arm_pose, joint_names=ARM_JOINTS
                    )
                wbc_updates += 1
                if wbc_updates % 50 == 0:
                    print(
                        f"[WH] wbc cmd=({cmd_vx:.2f},{cmd_vy:.2f},{cmd_wz:.2f})  "
                        f"updates={wbc_updates}  pelvis_z={robot_articulation.get_world_poses()[0][0][2]:.3f}"
                    )

            if arm_sub is not None:
                arm_sub.spin_once()
                arm_tgt = arm_sub.get_targets()
                if arm_tgt is not None:
                    robot_articulation.set_joint_position_targets(
                        arm_tgt[None, :], joint_names=ARM_JOINTS
                    )
                    if arm_sub.stats[0] == 1:
                        print("[WH] arm override    : EXTERNAL arm targets active (WBC legs/waist unaffected)")

            sent = g1.step(sim.current_time)
            if sent:
                scans += 1
                last_points = sent
                if scans % 10 == 0:
                    print(f"[WH] lidar          : scan {scans}, published {sent} pts")

            if step % 100 == 0:
                print(f"[WH] step {step:>6}  scans {scans}  points {last_points}")
            if args_cli.steps and step >= args_cli.steps:
                break
    except KeyboardInterrupt:
        print("\n[WH] interrupted")
    finally:
        if g1 is not None:
            import rclpy

            g1.destroy()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
