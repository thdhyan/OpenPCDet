"""Navigation presets 3-5 for ``g1_warehouse_sim.py --env`` (see
``g1_sim.environments`` for the hook contract). IRA owns the stage for these:
warehouse + walking people are set up by ``g1_sim.ira_actors`` before these
hooks run.

IRA bakes its navmesh before any of our props or the G1 exist, so people would
walk straight through them. ``build_props`` therefore carves navmesh holes
(NavMeshVolume "Exclude") at the robot spawn and at every box, then rebakes -
the characters only query the navmesh after ``timeline.play()``, so they route
around the new holes. The boxes themselves stay dynamic rigid bodies; the
navmesh ignores rigid bodies anyway (``excludeRigidBodies`` defaults on), which
is why the holes are explicit volumes rather than relying on the box geometry.
"""

from __future__ import annotations

from g1_sim.environments import spawn_box

# (center_xy, size_xy) of navmesh holes. G1 spawns at the origin facing +y.
ROBOT_FOOTPRINT = ((0.0, 0.0), (0.8, 0.8))
# path, edge length (m), floor xy, mass (kg). Off the front camera's line of
# sight to the robot (eye (0.4, 2.8)), inside the lidar/D435 field of view.
BIG_BOX = ("/World/Props/BigBox", 0.8, (0.9, 3.5), 10.0)
SMALL_BOX = ("/World/Props/SmallBox", 0.3, (-0.4, 4.5), 1.0)

# Env 5: box carried on the forearms (G1 link names, see the Dex3 USD).
FOREARM_LINKS = ("left_elbow_link", "right_elbow_link")
ELBOW_JOINTS = ("joints/left_elbow_joint", "joints/right_elbow_joint")
CARRY_BOX_SIZE = (0.24, 0.40, 0.18)  # robot-frame x (depth), y (width), z (height)
CARRY_BOX_POS = (0.22, 0.0, 0.26)    # centre in the pelvis frame at spawn: on top of both forearms
CARRY_BOX_MASS = 0.5


def _carve_navmesh_holes(stage, holes: dict[str, tuple]) -> None:
    """Navmesh "Exclude" volume (floor to 2 m) over each ``name: (center_xy,
    size_xy)`` footprint, then rebake and log the nearest walkable point to
    each hole centre (> half the footprint = people route around it). No-op
    without IRA (no navmesh, navigation extension possibly not even loaded)."""
    import time

    try:
        import omni.anim.navigation.core as nav
    except ImportError:
        nav = None
    inav = nav.acquire_interface() if nav else None
    if inav is None or inav.get_navmesh() is None:
        print("[NAV] no IRA navmesh (IRA off/failed) - navmesh holes skipped")
        return

    import carb
    import NavSchema
    import omni.kit.app
    from pxr import Gf, UsdGeom

    for name, (center_xy, size_xy) in holes.items():
        vol = NavSchema.NavMeshVolume.Define(stage, f"/World/NavExclude/{name}")
        vol.GetNavVolumeTypeAttr().Set("Exclude")
        UsdGeom.Boundable(vol.GetPrim()).GetExtentAttr().Set([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
        xf = UsdGeom.Xformable(vol.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(center_xy[0], center_xy[1], 0.75))
        xf.AddScaleOp().Set(Gf.Vec3f(size_xy[0], size_xy[1], 2.5))

    # Blocking bake: polling get_navmesh() doesn't work for a rebake, the old
    # navmesh stays valid until the new one replaces it.
    for _ in range(5):  # let the navigation plugin see the new volumes
        omni.kit.app.get_app().update()
    t0 = time.time()
    inav.start_navmesh_baking_and_wait()
    omni.kit.app.get_app().update()
    navmesh = inav.get_navmesh()
    if navmesh is None:
        raise RuntimeError("navmesh rebake failed")
    print(f"[NAV] navmesh rebaked with {len(holes)} holes ({time.time() - t0:.1f}s)")
    agent = nav.NavAgentDesc(radius=0.0, height=2.0, collision_gap=0.0)
    for name, ((x, y), _) in holes.items():
        p = navmesh.query_closest_point(carb.Float3(x, y, 0.0), agent=agent)[0]
        d = ((p[0] - x) ** 2 + (p[1] - y) ** 2) ** 0.5 if p is not None else float("nan")
        print(f"[NAV] {name:<14} centre ({x:+.2f}, {y:+.2f}): nearest walkable point {d:.2f} m away")


def _quatf(q):
    from pxr import Gf

    return Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary()))


def build_nav_people(stage) -> None:
    """Env 3: people route around the robot's spawn footprint."""
    _carve_navmesh_holes(stage, {"Robot": ROBOT_FOOTPRINT})


def build_nav_people_boxes(stage) -> None:
    """Env 4: one big and one small box in front of the robot, both navmesh holes."""
    holes = {"Robot": ROBOT_FOOTPRINT}
    for path, size, (x, y), mass in (BIG_BOX, SMALL_BOX):
        spawn_box(stage, path, size, (x, y, size / 2), mass=mass, color=(0.55, 0.40, 0.22))
        holes[path.rsplit("/", 1)[-1]] = ((x, y), (size, size))
        print(f"[ENV] box  {path:<34} {size:.2f} m  {mass:.1f} kg @ ({x:+.2f}, {y:+.2f})")
    _carve_navmesh_holes(stage, holes)


def attach_forearm_box(stage, robot_prim: str) -> None:
    """Env 5: light box fixed on top of both forearms, carried in front.

    * Placed at the spawn pose (elbows at the USD's initial angle, forearms
      ~horizontal) and welded with a FixedJoint to each forearm link, joint
      frames taken from that same pose, so nothing snaps at ``sim.reset()``.
    * The main loop's arm pose sends the elbows to -1.2 rad (hands up at face
      height), which would lift the box in front of the D435. The elbow lower
      limit is raised to the spawn angle instead, so that target just holds the
      forearms level against the limit (25 Nm drive cap, internal to the arm).
    * Both welds are excluded from the articulation (loop closure), and
      collisions between box and robot are filtered out (the box overlaps the
      hands); it still collides with the world and the people.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    pelvis_w = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{robot_prim}/pelvis"))
    box_w = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*CARRY_BOX_POS)) * pelvis_w
    rot_w = pelvis_w.ExtractRotation()
    box = spawn_box(stage, "/World/Props/CarriedBox", CARRY_BOX_SIZE, (0.0, 0.0, 0.0),
                    mass=CARRY_BOX_MASS, color=(0.55, 0.40, 0.22))
    xf = UsdGeom.Xformable(box)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(box_w.ExtractTranslation())
    xf.AddOrientOp().Set(_quatf(rot_w.GetQuat()))
    xf.AddScaleOp().Set(Gf.Vec3f(*CARRY_BOX_SIZE))
    robot_bodies = [p.GetPath() for p in Usd.PrimRange(stage.GetPrimAtPath(robot_prim)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    UsdPhysics.FilteredPairsAPI.Apply(box).CreateFilteredPairsRel().SetTargets(robot_bodies)

    box_nodes = Gf.Matrix4d().SetTransform(rot_w, box_w.ExtractTranslation())  # box frame without scale
    for link, joint in zip(FOREARM_LINKS, ELBOW_JOINTS):
        link_prim = stage.GetPrimAtPath(f"{robot_prim}/{link}")
        if not link_prim.IsValid():
            raise RuntimeError(f"forearm link {link_prim.GetPath()} not found")
        rel = box_nodes * cache.GetLocalToWorldTransform(link_prim).GetInverse()  # box pose in link frame
        fj = UsdPhysics.FixedJoint.Define(stage, f"/World/Props/CarriedBox_{link}_joint")
        fj.CreateBody0Rel().SetTargets([link_prim.GetPath()])
        fj.CreateBody1Rel().SetTargets([box.GetPath()])
        fj.CreateLocalPos0Attr().Set(Gf.Vec3f(rel.ExtractTranslation()))
        fj.CreateLocalRot0Attr().Set(_quatf(rel.ExtractRotationQuat()))
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        # Keep the welds out of the G1 articulation: otherwise PhysX pulls the
        # box in as a link and breaks the loop at right_shoulder_pitch_joint
        # (articulation came up 42 DOF, WBC disabled).
        fj.CreateExcludeFromArticulationAttr().Set(True)

        elbow = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{robot_prim}/{joint}"))
        spawn_deg = elbow.GetPrim().GetAttribute("state:angular:physics:position").Get()
        elbow.GetLowerLimitAttr().Set(float(spawn_deg))
        print(f"[ENV] carried box    : fixed to {link}, {joint.rsplit('/', 1)[-1]} lower limit -> {spawn_deg:.1f} deg")
    _dbg_box_watch(stage, robot_prim)  # TEMP
    print(f"[ENV] carried box    : {CARRY_BOX_SIZE} m, {CARRY_BOX_MASS} kg, robot collisions filtered ({len(robot_bodies)} bodies) @ {tuple(round(v, 3) for v in box_w.ExtractTranslation())}")



def _dbg_box_watch(stage, robot_prim):  # TEMP DEBUG
    import omni.kit.app
    from pxr import UsdGeom
    st = {"n": 0}
    def cb(_e):
        st["n"] += 1
        if st["n"] % 300:
            return
        c = UsdGeom.XformCache()
        b = c.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Props/CarriedBox")).ExtractTranslation()
        lw = c.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{robot_prim}/left_elbow_link")).ExtractTranslation()
        pv = c.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{robot_prim}/pelvis")).ExtractTranslation()
        print(f"[DBGBOX] upd {st['n']} box=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f}) box-lelbow=({b[0]-lw[0]:.3f},{b[1]-lw[1]:.3f},{b[2]-lw[2]:.3f}) pelvis_z={pv[2]:.3f}")
    global _DBG
    _DBG = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(cb)
