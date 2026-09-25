"""Env 6 (``--env social_static``): warehouse + named, static humans for the
LLM social-navigation evaluation.

Each human is an Isaac People character referenced at ``/World/Humans/<Name>``;
the prim name IS the person's name, so everything downstream (``/sim/humans``,
the LLM planner's prompt, the eval cameras) reads names back from USD instead
of keeping a second list. Names follow intent-sim's scenes.

The characters ship in T-pose and the stock idle clip targets a different
skeleton (it needs IRA's retargeting), so each one gets a one-frame
SkelAnimation built from its own rest pose with both upper arms rotated down.
No animation graph, no navmesh, no IRA: the humans cannot drift or fall.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

HUMANS_ROOT = "/World/Humans"
HUMANS_TOPIC = "/sim/humans"
LOCAL_PEOPLE = Path(__file__).resolve().parent.parent / "assets" / "people"

# name, Isaac People character, (x, y) [m], heading [deg, 0 = +x]. The G1 spawns
# at the origin facing +y: Dhyan stands alone ahead-right, Zach and Nirshal
# talk ahead-left (1.2 m apart, facing each other), Stephen is off to the right.
STATIC_HUMANS = [
    ("Dhyan", "M_Medical_01", (1.2, 3.6), -110.0),
    ("Zach", "male_adult_construction_05_new", (-2.4, 2.8), 0.0),
    ("Nirshal", "male_adult_police_04", (-1.2, 2.8), 180.0),
    ("Stephen", "F_Business_02", (3.0, 0.6), 160.0),
]
ARMS_DOWN_DEG = 80.0  # upper-arm rotation from T-pose toward the hips
COLLIDER_RADIUS = 0.25
COLLIDER_HEIGHT = 1.2  # capsule cylinder part; total height = this + 2 * radius


def character_usd(character: str) -> str:
    """Local mirror (scripts/fetch_people_assets.py) if present, else the Isaac asset root."""
    local = LOCAL_PEOPLE / "Characters" / character / f"{character}.usd"
    if local.exists():
        return str(local)
    from g1_sim.environments import asset_url

    return asset_url(f"Isaac/People/Characters/{character}/{character}.usd")


def _stand_pose(stage, human_path: str) -> None:
    """Bind a one-frame SkelAnimation: rest pose with the upper arms lowered."""
    from pxr import Gf, Usd, UsdSkel

    skel_prim = next((p for p in Usd.PrimRange(stage.GetPrimAtPath(human_path)) if p.IsA(UsdSkel.Skeleton)), None)
    if skel_prim is None:
        raise RuntimeError(f"no Skeleton under {human_path} (character reference unresolved?)")
    skel = UsdSkel.Skeleton(skel_prim)
    joints = [str(j) for j in skel.GetJointsAttr().Get()]
    local = [Gf.Matrix4d(m) for m in skel.GetRestTransformsAttr().Get()]
    index = {j: i for i, j in enumerate(joints)}

    world = []  # skeleton-space joint transforms (row vectors: child * parent)
    for i, j in enumerate(joints):
        world.append(local[i] * world[index[j.rsplit("/", 1)[0]]] if "/" in j else local[i])

    for side, sign in (("L", 1.0), ("R", -1.0)):  # T-pose arms lie along +/-x
        j = next(x for x in joints if x.endswith(f"/{side}_Upperarm"))
        i, parent = index[j], index[j.rsplit("/", 1)[0]]
        pivot = world[i].ExtractTranslation()
        about_pivot = (
            Gf.Matrix4d().SetTranslate(-pivot)
            * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), sign * ARMS_DOWN_DEG))
            * Gf.Matrix4d().SetTranslate(pivot)
        )
        local[i] = world[i] * about_pivot * world[parent].GetInverse()

    anim = UsdSkel.Animation.Define(stage, f"{human_path}/StandPose")
    anim.CreateJointsAttr(joints)
    anim.CreateTranslationsAttr([Gf.Vec3f(m.ExtractTranslation()) for m in local])
    anim.CreateRotationsAttr([Gf.Quatf(m.ExtractRotationQuat()) for m in local])
    anim.CreateScalesAttr([Gf.Vec3h(1, 1, 1)] * len(joints))
    UsdSkel.BindingAPI.Apply(skel_prim).CreateAnimationSourceRel().SetTargets([anim.GetPath()])


def build_social_static(stage) -> None:
    """Env 6: reference each character, pose it, and add a static capsule collider."""
    from pxr import Gf, UsdGeom, UsdPhysics

    from g1_sim.environments import _label

    UsdGeom.Scope.Define(stage, HUMANS_ROOT)
    for name, character, (x, y), heading in STATIC_HUMANS:
        path = f"{HUMANS_ROOT}/{name}"
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        prim.GetReferences().AddReference(character_usd(character))
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
        xf.AddRotateZOp().Set(heading + 90.0)  # characters face -y in their own frame
        _stand_pose(stage, path)

        cap = UsdGeom.Capsule.Define(stage, f"{path}/Collider")
        cap.CreateRadiusAttr(COLLIDER_RADIUS)
        cap.CreateHeightAttr(COLLIDER_HEIGHT)
        cap.CreateAxisAttr("Z")
        cap.CreatePurposeAttr(UsdGeom.Tokens.guide)  # physics only - invisible to cameras and RTX lidar
        cap.AddTranslateOp().Set(Gf.Vec3d(0, 0, COLLIDER_HEIGHT / 2 + COLLIDER_RADIUS))
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        _label(path, "person")
        print(f"[ENV] human {name:<8} {character:<32} @ ({x:+.2f}, {y:+.2f}) heading {heading:+.0f} deg")


def read_humans(stage) -> list[dict]:
    """Names + world poses of every child of /World/Humans, straight from USD."""
    from pxr import UsdGeom

    root = stage.GetPrimAtPath(HUMANS_ROOT)
    if not root.IsValid():
        return []
    cache = UsdGeom.XformCache()
    humans = []
    for prim in root.GetChildren():
        m = cache.GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        fwd = m.TransformDir((0.0, -1.0, 0.0))  # character forward is local -y
        humans.append({
            "name": prim.GetName(),
            "x": round(t[0], 3),
            "y": round(t[1], 3),
            "heading_deg": round(math.degrees(math.atan2(fwd[1], fwd[0])), 1),
        })
    return humans


_HUMANS_PUB = []  # keeps the rclpy node + update subscription alive


def attach_humans_publisher(stage, _robot_prim: str = "", period_s: float = 1.0) -> None:
    """Publish ``/sim/humans`` (std_msgs/String JSON, frame ``World``) at 1 Hz,
    transient-local so a planner started later still gets the roster."""
    import omni.kit.app
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import String

    humans = read_humans(stage)  # static: read once
    payload = json.dumps({"frame": "World", "humans": humans})
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("sim_humans")
    pub = node.create_publisher(String, HUMANS_TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    last = {"t": 0.0}

    def on_update(_event) -> None:
        if rclpy.ok() and time.monotonic() - last["t"] >= period_s:  # not after the sim's rclpy.shutdown()
            last["t"] = time.monotonic()
            pub.publish(String(data=payload))

    sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update, name="sim_humans")
    _HUMANS_PUB.append((node, pub, sub))
    print(f"[ENV] {HUMANS_TOPIC:<18}: {', '.join(h['name'] for h in humans)}")
