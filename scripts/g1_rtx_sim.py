#!/usr/bin/env python3
"""G1 with an RTX LiDAR Livox Mid-360, on Isaac Sim 6.0.

Run in the ``isaac`` conda env (Isaac Sim 6.0 / Python 3.12), not
``env_isaaclab`` (5.1 / 3.11):

    conda activate isaac
    python scripts/g1_rtx_sim.py                 # GUI, action graph visible
    python scripts/g1_rtx_sim.py --headless

Unlike the warp ray-caster in ``g1_ros2_sim.py``, RTX LiDAR ray-traces the
rendered scene, so it sees the robot itself and anything spawned later, and its
returns change as the robot moves.
"""

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser(description="G1 with RTX LiDAR Mid-360.")
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--num-prims",
    type=int,
    default=0,
    help="Use only the first N sensor prims. Fewer prims means a sparser sweep "
    "but less GPU load - try 2 if the GUI stutters on a small card.",
)
parser.add_argument("--steps", type=int, default=0, help="Stop after N steps; 0 runs forever.")
parser.add_argument("--no-ros2", action="store_true")
parser.add_argument("--no-camera", action="store_true", help="Skip the camera (saves render time).")
parser.add_argument(
    "--no-locomotion",
    action="store_true",
    help="Skip the decoupled_wbc cmd_vel bridge; robot just holds the USD's baked-in pose.",
)
parser.add_argument(
    "--config-dir",
    type=str,
    default="assets/lidar_configs_fast",
    help="Emitter-state profiles. 'fast' is 4 prims x 2 states (5.9 MB, "
    "quickest to load); 'light' is 4 x 5 (15 MB); the full set is 8 x 5 (30 MB). "
    "More states means a longer non-repetitive cycle but slower startup.",
)
parser.add_argument(
    "--use-og-helper",
    action="store_true",
    help="Publish via ROS2RtxLidarHelper instead of rclpy. It advertises the "
    "topic but does not emit on this setup; kept for GUI graph inspection.",
)
parser.add_argument(
    "--separate-topics",
    action="store_true",
    help="Publish each sensor prim on its own topic instead of merging them.",
)
parser.add_argument(
    "--no-room",
    action="store_true",
    help="Use flat ground instead of the Nucleus simple_room environment.",
)
args_cli = parser.parse_args()

SIM_RATE_HZ = 60.0

simulation_app = SimulationApp(
    {
        "headless": args_cli.headless,
        # Forces CPU-side buffering for LiDAR returns. The GPU path hits
        # cudaMemcpyAsync races when several RTX LiDAR prims publish over ROS2
        # (IsaacSim discussion #685).
        "/app/sensors/nv/lidar/outputBufferOnGPU": False,
        # Multi-tick (6.0 default) without motion BVH is unsupported - RTX
        # LiDAR reads fail with GMO magic-number errors. Must be command-line
        # args; config-dict keys apply too late to matter.
        "extra_args": [
            "--/renderer/raytracingMotion/enabled=true",
            "--/renderer/raytracingMotion/enableHydraEngineMasking=true",
            "--/renderer/raytracingMotion/enabledForHydraEngines=0,1,2,3",
        ],
    }
)

"""Rest everything follows."""

import carb
import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

from g1_sim.rtx_camera import (
    apply_semantics,
    attach_camera_publishers,
    attach_cmd_vel_subscriber,
    attach_robot_state_publishers,
    spawn_camera,
)
from g1_sim.rtx_lidar import (
    MID360_POS,
    attach_ros2_publishers,
    blind_radius,
    spawn_mid360,
)

ENABLE_ROS2 = not args_cli.no_ros2
ENABLE_LOCOMOTION = not args_cli.no_locomotion
WBC_CONTROL_HZ = 50.0  # decoupled_wbc's trained control rate (sim runs faster; see findings doc)

G1_USD = REPO / "assets/g1_29dof_sensors.usd"
ROBOT_PRIM = "/World/G1"
PEDESTRIANS = [(3.0, 0.0), (4.5, -2.0), (6.0, 2.5)]
SIMPLE_ROOM_USD = "/Isaac/Environments/Simple_Room/simple_room.usd"

# Loaded so the graph is inspectable in the GUI - Window > Visual Scripting >
# Action Graph - which is how you confirm what is publishing where.
GUI_EXTENSIONS = [
    "omni.graph.window.action",
    "omni.graph.window.generic",
    "omni.kit.widget.stage",
]


def enable_extensions() -> None:
    manager = omni.kit.app.get_app().get_extension_manager()

    # Must precede any rclpy import: the bridge also registers the ROS2
    # OmniGraph node types the publishers rely on.
    manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    manager.set_extension_enabled_immediate("isaacsim.sensors.rtx", True)

    if not args_cli.headless:
        for ext in GUI_EXTENSIONS:
            manager.set_extension_enabled_immediate(ext, True)

    for _ in range(20):
        omni.kit.app.get_app().update()


def _build_room(stage) -> None:
    """Load Nucleus simple_room; fall back to flat ground if unavailable."""
    try:
        import isaacsim.core.utils.stage as stage_utils

        stage_utils.add_reference_to_stage(
            usd_path=SIMPLE_ROOM_USD, prim_path="/World/SimpleRoom"
        )
        print(f"[RTX] environment    : {SIMPLE_ROOM_USD}")
        return
    except Exception as e:
        print(f"[RTX] Nucleus room not available ({e}), using flat ground")

    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.1))
    ground.AddScaleOp().Set(Gf.Vec3f(120.0, 120.0, 0.2))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    light = UsdLux.DistantLight.Define(stage, "/World/light")
    light.CreateIntensityAttr(3000.0)


def build_scene() -> None:
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")

    if not args_cli.no_room:
        _build_room(stage)
    else:
        ground = UsdGeom.Cube.Define(stage, "/World/ground")
        ground.CreateSizeAttr(1.0)
        ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.1))
        ground.AddScaleOp().Set(Gf.Vec3f(120.0, 120.0, 0.2))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

        light = UsdLux.DistantLight.Define(stage, "/World/light")
        light.CreateIntensityAttr(3000.0)

    for i, (x, y) in enumerate(PEDESTRIANS):
        box = UsdGeom.Cube.Define(stage, f"/World/targets/pedestrian_{i}")
        box.CreateSizeAttr(1.0)
        box.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.875))
        box.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 1.75))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())

    if not G1_USD.exists():
        raise SystemExit(
            f"[RTX] {G1_USD} not found - run scripts/convert_g1_urdf_to_usd.py first"
        )
    robot = stage.DefinePrim(ROBOT_PRIM, "Xform")
    robot.GetReferences().AddReference(str(G1_USD))
    xform = UsdGeom.Xformable(robot)
    translate = next(
        (op for op in xform.GetOrderedXformOps() if "translate" in op.GetOpName()), None
    )
    if translate is None:
        translate = xform.AddTranslateOp()
    translate.Set(Gf.Vec3d(0.0, 0.0, 0.8))


def main() -> None:
    from isaacsim.core.api import SimulationContext

    enable_extensions()
    build_scene()

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / SIM_RATE_HZ,
        rendering_dt=1.0 / SIM_RATE_HZ,
    )

    # Register the articulation with physics. Without this the joints never
    # move, so TF and joint states report a frozen pose even when the graphs
    # are wired correctly. The root carries ArticulationRootAPI on the pelvis.
    from isaacsim.core.prims import Articulation

    robot_articulation = Articulation(f"{ROBOT_PRIM}/pelvis", name="g1")
    sim.reset()
    print(f"[RTX] articulation   : {robot_articulation.num_dof} DOF")

    wbc_bridge = None
    if ENABLE_LOCOMOTION:
        from g1_sim.wbc_bridge import (
            ARM_JOINTS,
            ARM_KD,
            ARM_KP,
            ALL_JOINTS,
            KD,
            KP,
            LEG_WAIST_JOINTS,
            WbcBridge,
            quat_rotate_inverse,
        )

        dof_names = set(robot_articulation.dof_names or [])
        missing = [j for j in ALL_JOINTS if j not in dof_names]
        if missing:
            print(f"[RTX] WBC bridge     : DISABLED - USD is missing joints {missing}")
        else:
            wbc_bridge = WbcBridge(
                REPO / "assets/policy/GR00T-WholeBodyControl-Balance.onnx",
                REPO / "assets/policy/GR00T-WholeBodyControl-Walk.onnx",
            )
            # The USD's baked-in drive gains (uniform 100/10 from
            # convert_g1_urdf_to_usd.py) don't match what the policy was
            # trained with - overwrite them per-joint before the first step.
            robot_articulation.set_gains(
                kps=KP[None, :], kds=KD[None, :], joint_names=LEG_WAIST_JOINTS
            )
            robot_articulation.set_gains(
                kps=np.full((1, len(ARM_JOINTS)), ARM_KP, dtype=np.float32),
                kds=np.full((1, len(ARM_JOINTS)), ARM_KD, dtype=np.float32),
                joint_names=ARM_JOINTS,
            )
            print("[RTX] WBC bridge     : loaded (decoupled_wbc Balance/Walk policies, gains overridden)")

    # Mount ON the mid360_link prim, which already carries the URDF's
    # torso->sensor pose. Identity local transform means the returns come out in
    # the exact frame published as `mid360_link`, so frame_id and point origin
    # are the same prim (see g1_warehouse_sim.py for the below-ground bug this
    # fixed). mid360_link is rigidly fixed to torso, so it still inherits the
    # torso's motion.
    mount = f"{ROBOT_PRIM}/mid360_link"
    if not omni.usd.get_context().get_stage().GetPrimAtPath(mount).IsValid():
        raise SystemExit(f"[RTX] mount prim {mount} missing from the USD")

    prim_paths = spawn_mid360(
        mount,
        config_dir=REPO / args_cli.config_dir,
        translation=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    if args_cli.num_prims and args_cli.num_prims < len(prim_paths):
        prim_paths = prim_paths[: args_cli.num_prims]

    print(f"[RTX] sensor prims   : {len(prim_paths)}")
    for p in prim_paths:
        print(f"[RTX]   {p}")

    mount_height = 0.8 + MID360_POS[2]
    print(f"[RTX] mount height   : {mount_height:.2f} m")
    print(f"[RTX] blind radius   : {blind_radius(mount_height):.2f} m (no nadir ray)")

    publisher = None
    cmd_vel_graph_path = None
    if ENABLE_ROS2:
        # The OmniGraph helper is always built so the graph is visible in the
        # GUI, but by default the points are published by rclpy reading the
        # sensor annotator - the helper advertises without emitting.
        graph = attach_ros2_publishers(
            prim_paths,
            sim_rate_hz=SIM_RATE_HZ,
            combine=not args_cli.separate_topics,
        )
        print(f"[RTX] lidar graph    : {graph}")

        # TF, joint states and /clock - what RViz needs to draw the robot.
        state_graph = attach_robot_state_publishers(ROBOT_PRIM)
        print(f"[RTX] state graph    : {state_graph}  (/tf, /g1/joint_states, /clock)")

        if wbc_bridge is not None:
            cmd_vel_graph_path = attach_cmd_vel_subscriber()
            print(f"[RTX] cmd_vel graph  : {cmd_vel_graph_path}  (/g1/cmd_vel)")

        if not args_cli.no_camera:
            camera_prim = spawn_camera(f"{ROBOT_PRIM}/torso_link")
            cam_graph = attach_camera_publishers(camera_prim)
            print(f"[RTX] camera graph   : {cam_graph}")
            print("[RTX] camera topics  : /g1/camera/{rgb,depth,semantic,camera_info}")

            # Semantic segmentation only reports labelled prims, so the
            # targets need classes or the image is entirely background.
            labels = {f"/World/targets/pedestrian_{i}": "pedestrian" for i in range(len(PEDESTRIANS))}
            labels["/World/ground"] = "ground"
            labels["/World/SimpleRoom"] = "ground"
            labels[ROBOT_PRIM] = "robot"
            print(f"[RTX] semantics      : {apply_semantics(labels)} prims labelled")

        if not args_cli.use_og_helper:
            import rclpy

            from g1_sim.rtx_publisher import RtxLidarPublisher

            rclpy.init()
            publisher = RtxLidarPublisher(prim_paths, publish_rate=10.0)
            print("[RTX] publisher      : rclpy (annotator)")
        else:
            print("[RTX] publisher      : OmniGraph helper")
    else:
        print("[RTX] ROS2 disabled")

    # OnPlaybackTick - which drives the ROS2 helpers - only fires while the
    # timeline is playing. Stepping physics alone leaves the graph dormant and
    # the topic advertised but silent.
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print(f"[RTX] timeline       : playing={timeline.is_playing()}")
    print("[RTX] running\n")

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

            if wbc_bridge is not None and sim.current_time >= wbc_next_time:
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
                    # Arms aren't policy-controlled; hold them at the URDF
                    # zero pose, same as run_mujoco_gear_wbc.py's fixed PD.
                    robot_articulation.set_joint_position_targets(
                        np.zeros((1, len(ARM_JOINTS)), dtype=np.float32), joint_names=ARM_JOINTS
                    )
                wbc_updates += 1
                if wbc_updates % 50 == 0:
                    print(
                        f"[RTX] wbc cmd=({cmd_vx:.2f},{cmd_vy:.2f},{cmd_wz:.2f})  "
                        f"updates={wbc_updates}  pelvis_z={robot_articulation.get_world_poses()[0][0][2]:.3f}"
                    )

            if publisher is not None:
                # Every step contributes its slice of the sweep; publish() then
                # emits the assembled scan at the sensor's rate.
                publisher.accumulate()
                # Stamp from the simulator's own clock so the cloud agrees
                # with /clock and with the TF the action graph publishes.
                sent = publisher.publish(sim.current_time)
                if sent:
                    scans += 1
                    last_points = sent
                publisher.spin_once()

            if step % 100 == 0:
                print(f"[RTX] step {step:>6}  scans {scans}  points {last_points}")
            if args_cli.steps and step >= args_cli.steps:
                break
    except KeyboardInterrupt:
        print("\n[RTX] interrupted")
    finally:
        if publisher is not None:
            import rclpy

            publisher.destroy()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
