"""Optional humans + Nova Carters via Isaac Replicator Agent (IRA), for the
warehouse SDG scene.

IRA (``isaacsim.replicator.agent.core``, extension version 1.6.8 on this
install - noticeably different API from the 5.1.0-era docs/``sdg_scheduler.py``
CLI those docs describe) is entirely config-driven: build a YAML matching its
``RootConfig`` schema, then call two async entry points:

    isaacsim.replicator.agent.core.api.load_config_file(path)
    await isaacsim.replicator.agent.core.api.setup_simulation()

``setup_simulation()`` opens the environment USD **as the new root stage**
(``StageManager.wait_for_stage_open``, a full stage replace - not a
reference), then loads characters, then robots, baking a navmesh in between.
Because it owns the stage, IRA must run *before* anything else is added to
the scene - the caller adds G1 and its sensors afterward, on the resulting
stage.

**Real bug found and worked around**: ``EnvironmentLoader.load()`` calls
``omni.metropolis.pipeline.simulation_util.ensure_navmesh_ready()``, which
polls ``inav.get_navmesh()`` for a **hard-coded 100 frames** with no config
knob to raise it. On the real ``warehouse.usd`` (291 meshes) baking
genuinely succeeds but needs ~600-700 frames (~7-8s headless on this
machine), so every attempt failed with "NavMesh building failed after 101
frames" even though baking was progressing fine. Confirmed by direct
`omni.anim.navigation.core` testing outside IRA: ``start_navmesh_baking()``
can only be made to return ``True`` via IRA's own pipeline (a standalone
reference-add of the warehouse USD returns ``can_bake=False`` - IRA's
environment/collider setup does something extra first that makes baking
possible at all), and once baking is running it finishes in under 700
frames, comfortably inside a raised budget. ``environment_loader.py`` does
``from ...simulation_util import ensure_navmesh_ready``, which copies the
name into its own module namespace, so the fix has to patch that module's
attribute specifically - patching the origin module has no effect.
"""

from __future__ import annotations

import time
from pathlib import Path

DEFAULT_MAX_NAVMESH_FRAMES = 3000  # ~30x the stock cap; real bake measured ~700 frames

WAREHOUSE_REL = "Isaac/Environments/Simple_Warehouse/warehouse.usd"  # IRA-relative form (no leading slash)

# Human spawn points (x, y, z) around the G1 spawn at the origin, so people walk
# through the sensors' and capture views' range instead of anywhere in the
# warehouse. IRA snaps each onto the navmesh; clear of the robot and the env-4
# boxes (g1_sim.nav_environments).
HUMAN_SPAWNS = [
    (3.0, 1.5, 0.0), (-3.0, 2.5, 0.0), (2.5, -2.5, 0.0),
    (-2.5, -3.0, 0.0), (1.0, 5.0, 0.0), (-4.5, 0.0, 0.0),
]


def write_config(
    path: Path,
    *,
    warehouse_rel: str = WAREHOUSE_REL,
    num_humans: int = 2,
    num_carters: int = 1,
    seed: int = 42,
    duration_s: float = 200.0,
) -> Path:
    """Write a minimal IRA YAML config: N wandering characters + N wandering
    Nova Carters in the given warehouse. No ``sensor``/``replicator`` section
    - we don't want IRA's own writer-driven capture loop, only the actors it
    spawns; our own G1 sensors do the real ROS2 publishing.
    """
    import yaml

    config = {
        "isaacsim.replicator.agent": {
            # IRA 1.7.x rejects < 1.7.0 ("no migration was provided") and
            # setup then silently falls back to a people-free warehouse.
            "version": "1.7.0",
            "seed": seed,
            "simulation_duration": duration_s,
            "environment": {"base_stage_asset_path": warehouse_rel},
        }
    }
    if num_humans > 0:
        config["isaacsim.replicator.agent"]["character"] = {
            "groups": {
                "humans": {
                    "num": num_humans,
                    "asset_path": "Isaac/People/Characters/",
                    "spawn_positions": [list(p) for p in HUMAN_SPAWNS[:num_humans]],
                    "routines": [
                        {
                            "wander": {
                                "weight": 1.0,
                                "repeat": 1,
                                "walk": {
                                    "speed_range": [0.8, 1.2],
                                    "distance_range": [3.0, 8.0],
                                    "navigation_areas": [],
                                },
                                "idle": [
                                    {"animation": "idle", "weight": 1.0, "time_range": [2.0, 4.0]}
                                ],
                            }
                        }
                    ],
                }
            }
        }
    if num_carters > 0:
        config["isaacsim.replicator.agent"]["robot"] = {
            "groups": {
                "carters": {
                    "num": num_carters,
                    "config_file_path": "nova_carter.yaml",
                    "routines": [
                        {
                            "wander": {
                                "weight": 1.0,
                                "repeat": 1,
                                "move": {"distance_range": [5.0, 10.0], "navigation_areas": []},
                                "idle": {"time_range": [2.0, 4.0]},
                            }
                        }
                    ],
                }
            }
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def enable_extension() -> None:
    """Enable the IRA core extension. Must happen before importing its api
    module, same requirement as the ROS2 bridge extension elsewhere in this
    codebase."""
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.replicator.agent.core", True)
    for _ in range(30):
        omni.kit.app.get_app().update()


def _patch_navmesh_frame_budget(max_frames: int = DEFAULT_MAX_NAVMESH_FRAMES) -> None:
    """See module docstring - raises the hard-coded 100-frame navmesh bake
    poll cap that makes ``EnvironmentLoader.load()`` fail on real warehouse
    meshes. Patches the name as imported into
    ``isaacsim.replicator.agent.core.scene_assembly.environment_loader``,
    not the origin ``omni.metropolis.pipeline.simulation_util`` module."""
    import carb
    import omni.kit.app
    import isaacsim.replicator.agent.core.scene_assembly.environment_loader as environment_loader

    async def _ensure_navmesh_ready_patched() -> bool:
        import omni.anim.navigation.core as nav

        inav = nav.acquire_interface()
        if inav is None:
            carb.log_error("[ira_actors] NavMesh interface not available")
            return False
        if inav.get_navmesh() is not None:
            return True
        if not inav.start_navmesh_baking():
            carb.log_error("[ira_actors] start_navmesh_baking() returned False")
            return False
        frame_count = 0
        t0 = time.time()
        while inav.get_navmesh() is None:
            await omni.kit.app.get_app().next_update_async()
            frame_count += 1
            if frame_count > max_frames:
                carb.log_error(
                    f"[ira_actors] navmesh bake exceeded raised budget of {max_frames} frames "
                    f"({time.time()-t0:.1f}s) - stock IRA cap is 100, this needs raising further"
                )
                return False
        print(f"[ira_actors] navmesh baked in {frame_count} frames ({time.time()-t0:.1f}s)")
        return True

    environment_loader.ensure_navmesh_ready = _ensure_navmesh_ready_patched


async def setup(
    config_path: Path,
    *,
    max_navmesh_frames: int = DEFAULT_MAX_NAVMESH_FRAMES,
) -> bool:
    """Load ``config_path`` and run IRA's full setup (environment + navmesh +
    characters + robots) on the current stage - which becomes a *new* stage,
    replacing whatever was open before. Returns ``True`` on success, ``False``
    on any failure (config validation, navmesh bake, asset load) so the
    caller can fall back to a plain warehouse load with no actors rather than
    crashing the whole pipeline over an IRA-specific failure.
    """
    from isaacsim.replicator.agent.core import api as ira_api

    _patch_navmesh_frame_budget(max_navmesh_frames)

    if not ira_api.load_config_file(str(config_path)):
        print(f"[ira_actors] FAILED to load config {config_path}")
        return False

    try:
        await ira_api.setup_simulation()
    except Exception as e:
        print(f"[ira_actors] FAILED setup_simulation(): {e!r}")
        return False

    return True


def discover_prims_at(stage, scope_path: str) -> list[str]:
    """Return the direct children of ``scope_path``, or ``[]`` if it doesn't
    exist (e.g. the carters scope when ``--num-carters 0``)."""
    prim = stage.GetPrimAtPath(scope_path)
    if not prim.IsValid():
        return []
    return [child.GetPath().pathString for child in prim.GetChildren()]


def strip_carter_cameras(stage, carters_scope: str = "/World/Robots/carters") -> int:
    """Deactivate every camera on the Nova Carters, keeping the robots themselves.

    Each Nova Carter ships a Hawk/Owl sensor rig (~12 cameras under
    ``chassis_link/sensors/``). We never publish them - our own G1 D435 does the
    camera work - but Hydra still processes each one, and on an 8 GB GPU the
    stack of RTX camera passes is enough to OOM-crash the whole app. Setting the
    camera prims inactive removes them from stage traversal (so Hydra never
    allocates a render pass for them) while the carter body, physics, IMU and
    IRA-driven motion are untouched.

    Returns the number of cameras deactivated.
    """
    from pxr import Usd, UsdGeom

    scope = stage.GetPrimAtPath(carters_scope)
    if not scope.IsValid():
        return 0

    n = 0
    for prim in Usd.PrimRange(scope):
        if prim.IsA(UsdGeom.Camera):
            prim.SetActive(False)
            n += 1
    return n


def discover_actor_prims(
    stage,
    humans_scope: str = "/World/Characters/humans",
    carters_scope: str = "/World/Robots/carters",
) -> list[str]:
    """Return the root Xform of every spawned human and Nova Carter.

    IRA names actors ``humans_0``, ``humans_1``, ... and (for a single
    carter) plain ``Nova_Carter`` with no index suffix, so the count-based
    naming isn't predictable from ``--num-humans``/``--num-carters`` alone -
    read it back from the stage instead of guessing the pattern.
    """
    return discover_prims_at(stage, humans_scope) + discover_prims_at(stage, carters_scope)


def find_imu_prim(stage, root_path: str):
    """Find an existing IMU sensor prim already under ``root_path``.

    Nova Carter comes from IRA with its own IMU already set up (per NVIDIA's
    Replicator Agent robot-properties docs) - creating a *second* one via
    ``IMU.create()`` applies a physics schema to the rigid body IRA's own
    wander controller already holds a tensor view into, which invalidates
    that view and spams "Simulation view object is invalidated" on every
    tick thereafter (confirmed live with 4 carters, 2026-08-10). Read the
    stock sensor instead of authoring a new one.
    """
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        type_name = prim.GetTypeName()
        if "imu" in type_name.lower() or "imu" in prim.GetName().lower():
            return prim.GetPath().pathString
    return None


def attach_carter_imu_publishers(
    stage,
    carter_prim_paths: list[str],
) -> list[str]:
    """Publish each Nova Carter's existing IMU sensor (IRA's own, not one we
    create) as ``sensor_msgs/Imu`` on ``/carter_N/imu``."""
    from g1_sim.rtx_camera import attach_imu_publisher

    graph_paths = []
    for i, carter_path in enumerate(carter_prim_paths):
        imu_prim = find_imu_prim(stage, carter_path)
        if imu_prim is None:
            print(f"[ira_actors] carter {i} IMU  : not found under {carter_path}, skipping")
            continue
        topic = f"/carter_{i}/imu"
        graph_path = attach_imu_publisher(
            imu_prim,
            graph_path=f"/ActionGraph/CarterImuROS2_{i}",
            topic=topic,
            frame_id=f"carter_{i}_imu",
        )
        graph_paths.append(graph_path)
        print(f"[ira_actors] carter {i} IMU  : {topic}  prim={imu_prim}")
    return graph_paths


_ACTOR_TF = []  # keeps the rclpy node + update subscription alive


def attach_actor_tf_publishers(
    actor_prim_paths: list[str],
    topic_name: str = "/tf",
) -> str | None:
    """Publish one TF frame per human/Nova Carter (root pose, parent "World",
    child = the actor's prim name, e.g. ``humans_0``), stamped with sim time.

    Poses come from IRA's runtime agents (``AgentsManager``, created on
    ``timeline.play()``), not from USD: the behavior system moves characters
    in Fabric only, so the actor's USD Xform *and* its SkelRoot stay at the
    spawn point (verified 2026-09-24: USD frozen while the agent walked
    metres). The previous OmniGraph ``ROS2PublishTransformTree`` on those
    prims therefore published frozen poses. Runs from the app update stream,
    so it needs no hook in the main loop.
    """
    if not actor_prim_paths:
        return None

    import omni.kit.app
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.metropolis.pipeline.agent import AgentsManager
    from tf2_msgs.msg import TFMessage

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("ira_actor_tf")
    pub = node.create_publisher(TFMessage, topic_name, 10)
    frames: dict[str, str] = {}  # agent prim path -> actor frame name
    last = {"t": -1.0}

    def on_update(_event) -> None:
        t = SimulationManager.get_simulation_time()
        if t == last["t"]:
            return
        last["t"] = t
        msg = TFMessage()
        for agent in AgentsManager.get_instance().get_runtime_agent_instances():
            key = agent.prim.GetPath().pathString
            if key not in frames:  # agent prim is nested under the actor root
                frames[key] = next((a.rsplit("/", 1)[-1] for a in actor_prim_paths if key.startswith(a + "/") or key == a), "")
            pos, rot = agent.get_world_position(), agent.get_world_rotation()
            if not frames[key] or pos is None or rot is None:
                continue
            tf = TransformStamped()
            tf.header.stamp.sec = int(t)
            tf.header.stamp.nanosec = int((t - int(t)) * 1e9)
            tf.header.frame_id = "World"
            tf.child_frame_id = frames[key]
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = pos[0], pos[1], pos[2]
            q = tf.transform.rotation
            q.x, q.y, q.z, q.w = rot[0], rot[1], rot[2], rot[3]  # carb.Float4 is xyzw
            msg.transforms.append(tf)
        if msg.transforms:
            pub.publish(msg)

    sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update, name="ira_actor_tf")
    _ACTOR_TF.append((node, pub, sub))
    return f"rclpy {topic_name} (World -> {', '.join(a.rsplit('/', 1)[-1] for a in actor_prim_paths)})"


def run_setup_blocking(
    simulation_app,
    config_path: Path,
    *,
    max_navmesh_frames: int = DEFAULT_MAX_NAVMESH_FRAMES,
    timeout_s: float = 240.0,
) -> bool:
    """Synchronous wrapper: pumps ``simulation_app.update()`` until
    :func:`setup` (a coroutine) finishes or ``timeout_s`` elapses. Standalone
    Isaac Sim scripts have no running asyncio event loop of their own, so
    this is the same "ensure_future + drive the app manually" pattern used
    elsewhere for one-off async Isaac Sim calls.
    """
    import asyncio

    task = asyncio.ensure_future(setup(config_path, max_navmesh_frames=max_navmesh_frames))
    t0 = time.time()
    last_log = t0
    while not task.done():
        simulation_app.update()
        if time.time() - last_log > 10.0:
            print(f"[ira_actors] setup_simulation still running ({time.time()-t0:.1f}s)")
            last_log = time.time()
        if time.time() - t0 > timeout_s:
            print(f"[ira_actors] FAILED: setup exceeded {timeout_s}s timeout")
            return False

    exc = task.exception()
    if exc is not None:
        print(f"[ira_actors] FAILED: {exc!r}")
        return False

    ok = task.result()
    print(f"[ira_actors] setup {'OK' if ok else 'FAILED'} in {time.time()-t0:.1f}s")
    return ok


def save_baked_scene(stage, path: Path) -> None:
    """Export the current stage (post-IRA-setup: warehouse + navmesh +
    spawned characters/carters) to a standalone USD file, so a future run can
    skip the Nucleus warehouse download and the navmesh bake by loading this
    directly instead of calling :func:`setup` again.

    Caveat: this captures composed geometry/prim state as of the save
    moment, not IRA's live Python wander controllers - characters/carters
    loaded back from this file sit static at their saved pose until IRA's
    own ``setup_simulation()`` runs again. Useful for iterating on the
    warehouse/lidar/robot pipeline without re-paying the ~60-80s bake cost
    each launch; not a substitute for a real run when actor motion matters.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stage.Export(str(path))
    print(f"[ira_actors] baked scene saved -> {path}")


def load_baked_scene(path: Path):
    """Open a previously :func:`save_baked_scene`'d USD as the new root
    stage - mirrors IRA's own ``setup_simulation()`` behaviour of replacing
    the root stage (not adding a reference), so downstream code (e.g.
    ``spawn_g1``'s session-layer edit-target logic) doesn't need to know
    which path built the scene."""
    import omni.usd

    ctx = omni.usd.get_context()
    ctx.open_stage(str(path))
    print(f"[ira_actors] baked scene loaded <- {path}")
    return ctx.get_stage()
