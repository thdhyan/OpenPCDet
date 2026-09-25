"""Scene presets for ``scripts/g1_warehouse_sim.py --env <name>``.

Every preset uses the same G1 (Dex3, Mid-360, D435, IMUs) and the same WBC
pipeline; a preset only decides

* whether IRA owns the stage (``ira=True``: warehouse + walking people via
  ``g1_sim.ira_actors``; otherwise the static warehouse fallback),
* which props go into the stage before the robot (``build_props``), and
* what gets attached to the robot after it is loaded (``post_robot``).

Both hooks run before ``sim.reset()``. The robot always spawns at the origin
with yaw +90 deg, i.e. facing +y (IsaacLab locomanip pose).

Navigation presets (3-5) live in ``g1_sim.nav_environments``; the named static
humans of preset 6 in ``g1_sim.social_environments``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# Table + steering wheel poses from IsaacLab
# isaaclab_tasks/contrib/locomanip_pick_place/locomanipulation_g1_env_cfg.py
TABLE_PATH = "/World/Props/PackingTable"
TABLE_USD = "Isaac/Props/PackingTable/packing_table.usd"
TABLE_POS = (0.0, 0.55, -0.3)
# ISAACLAB_NUCLEUS_DIR is {root}/Isaac/IsaacLab. The old path dropped the
# "Isaac/" segment -> 404 -> empty reference -> invisible wheel.
WHEEL_USD = "Isaac/IsaacLab/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd"
WHEEL_POS = (-0.35, 0.45, 0.6996)
WHEEL_SCALE = 0.75


SURFACE_GAP = 0.003  # drop height for props placed on a surface [m]
# Packing-table top (SM_HeavyDutyPackingTable_C02_01 bbox max z, measured
# 2026-09-24); the asset's overall bbox top is higher because of the tray.
TABLE_SURFACE_Z = 0.694


@dataclass(frozen=True)
class EnvSpec:
    description: str
    ira: bool = False
    num_humans: int = 0
    build_props: Callable | None = None  # (stage) -> None
    post_robot: Callable | None = None  # (stage, robot_prim_path) -> None
    static_targets: bool = True  # fallback scene adds the LiDAR test cubes (g1_warehouse_sim.PEDESTRIANS)


def asset_url(rel: str) -> str:
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    if not root:
        raise RuntimeError("no Isaac asset root (Nucleus/S3) reachable")
    return f"{root}/{rel.lstrip('/')}"


def _set_pose(prim, pos, rpy_deg=(0.0, 0.0, 0.0), scale=None) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rpy_deg))
    if scale is not None:
        s = (scale,) * 3 if isinstance(scale, (int, float)) else scale
        xf.AddScaleOp().Set(Gf.Vec3f(*s))


def rigid_bodies(prim) -> list:
    from pxr import Usd, UsdPhysics

    return [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def make_rigid(prim, *, mass: float | None = None, kinematic: bool = False) -> None:
    """Like IsaacLab's ``rigid_props``: if the asset already has rigid bodies,
    only set kinematic/mass on them. Adding another body on the root would
    nest bodies (the old steering-wheel/table bug). Visual-only assets (YCB
    Axis_Aligned, ...) get one body on the root plus convex-hull colliders."""
    from pxr import Usd, UsdGeom, UsdPhysics

    bodies = rigid_bodies(prim)
    if not bodies:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        bodies = [prim]
        if not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim)):
            for p in Usd.PrimRange(prim):
                if p.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(p)
                    UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
    for b in bodies:
        UsdPhysics.RigidBodyAPI(b).CreateKinematicEnabledAttr(kinematic)
    if mass is not None:
        for b in bodies:
            UsdPhysics.MassAPI.Apply(b).CreateMassAttr(mass / len(bodies))


def spawn_usd_prop(stage, path: str, usd_rel: str, pos, *, rpy_deg=(0.0, 0.0, 0.0), scale=None,
                   mass: float | None = None, kinematic: bool = False, label: str | None = None,
                   on_surface: float | None = None):
    """Pose goes on a fresh Xform at ``path``; the asset is referenced into
    ``path/asset`` so its own root xform ops (often double-precision) stay
    untouched. Physics goes on the asset root. ``on_surface``: shift so the
    asset's bbox bottom sits just above that z and its bbox centre is at
    ``pos`` x/y. ``label``: semantic class (segmentation, VLM caption).
    Returns the asset prim."""
    holder = stage.DefinePrim(path, "Xform")
    _set_pose(holder, pos, rpy_deg, scale)
    prim = stage.DefinePrim(f"{path}/asset", "Xform")
    prim.GetReferences().AddReference(asset_url(usd_rel))
    n_children = len(prim.GetChildren())
    print(f"[ENV] prop {path:<34} {usd_rel.rsplit('/', 1)[-1]:<28} children={n_children}")
    if n_children == 0:
        # An unresolvable reference is only a warning in USD - fail loudly.
        raise RuntimeError(f"{path}: reference {usd_rel} resolved to nothing")
    if on_surface is not None:
        lo, hi = _world_range(stage, path)
        c = (lo + hi) / 2
        _set_pose(holder, (2 * pos[0] - c[0], 2 * pos[1] - c[1], pos[2] + on_surface + SURFACE_GAP - lo[2]), rpy_deg, scale)
    make_rigid(prim, mass=mass, kinematic=kinematic)
    if label:
        _label(path, label)
    return prim


def _world_range(stage, path: str):
    from pxr import Usd, UsdGeom

    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return r.GetMin(), r.GetMax()


def _label(path: str, label: str) -> None:
    from g1_sim.rtx_camera import apply_semantics

    apply_semantics({path: label})


def spawn_box(stage, path: str, size, pos, *, mass: float = 0.2, color=(0.8, 0.1, 0.1), kinematic=False,
              label: str | None = None):
    """Dynamic box. ``size`` is edge length (float) or (x, y, z) in metres."""
    from pxr import Gf, UsdGeom

    size = (size,) * 3 if isinstance(size, (int, float)) else size
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _set_pose(cube.GetPrim(), pos, scale=size)
    from pxr import UsdPhysics

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    rb.CreateKinematicEnabledAttr(kinematic)
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(mass)
    if label:
        _label(path, label)
    return cube.GetPrim()


def add_table(stage):
    # kinematic=True reaches only the asset's own bodies (the tray); the table
    # mesh is a plain static collider, same as IsaacLab.
    return spawn_usd_prop(stage, TABLE_PATH, TABLE_USD, TABLE_POS, kinematic=True, label="table")


def log_prop_poses(root: str = "/World/Props") -> None:
    """Print PhysX world position of every rigid body under ``root``."""
    import omni.physx
    import omni.usd

    physx = omni.physx.get_physx_interface()
    for b in rigid_bodies(omni.usd.get_context().get_stage().GetPrimAtPath(root)):
        t = physx.get_rigidbody_transformation(str(b.GetPath()))
        if t.get("ret_val"):
            x, y, z = t["position"]
            print(f"[ENV] pose {str(b.GetPath()):<60} ({x:+.3f}, {y:+.3f}, {z:+.3f})")


# --- env 1 --------------------------------------------------------------------


def build_tabletop_wheel(stage) -> None:
    add_table(stage)
    spawn_usd_prop(stage, "/World/Props/SteeringWheel", WHEEL_USD, WHEEL_POS, scale=WHEEL_SCALE, mass=0.5,
                   label="steering wheel")


# --- env 2 --------------------------------------------------------------------

# Keep env 2 deliberately small: one large red target and one small green
# target. The table is the only supporting prop.
CLUSTER_CUBES = [
    # name, size (m), (x, y), rgb, mass (kg), semantic label
    ("LargeRedCube", (0.10, 0.10, 0.10), (-0.15, 0.37), (0.85, 0.08, 0.08), 0.12, "large red cube"),
    ("SmallGreenCube", (0.05, 0.05, 0.05), (-0.04, 0.44), (0.08, 0.65, 0.15), 0.03, "small green cube"),
]


def _spawn_tabletop_cubes(stage) -> None:
    for name, size, (x, y), rgb, mass, label in CLUSTER_CUBES:
        spawn_box(
            stage,
            f"/World/Props/{name}",
            size,
            (x, y, TABLE_SURFACE_Z + size[2] / 2 + SURFACE_GAP),
            mass=mass,
            color=rgb,
            label=label,
        )


def build_tabletop_cluster(stage) -> None:
    add_table(stage)
    _spawn_tabletop_cubes(stage)


def build_tabletop_cluster_local(stage) -> None:
    """Build the env-2 table and two-cube layout without remote assets."""
    spawn_box(stage, "/World/Props/PackingTable", (1.20, 0.80, 0.08),
              (0.0, 0.55, TABLE_SURFACE_Z - 0.04), mass=1.0,
              color=(0.48, 0.30, 0.16), kinematic=True, label="table")
    _spawn_tabletop_cubes(stage)


# --- env 3-5 (implemented in g1_sim.nav_environments) ----------------------


def _nav(fn_name: str) -> Callable:
    def hook(*args):
        from g1_sim import nav_environments

        return getattr(nav_environments, fn_name)(*args)

    return hook


def _social(fn_name: str) -> Callable:
    def hook(*args):
        from g1_sim import social_environments

        return getattr(social_environments, fn_name)(*args)

    return hook


ENVIRONMENTS: dict[str, EnvSpec] = {
    "tabletop_wheel": EnvSpec("1. packing table + steering wheel", build_props=build_tabletop_wheel),
    "tabletop_cluster": EnvSpec(
        "2. packing table + large red cube + small green cube",
        build_props=build_tabletop_cluster,
    ),
    "nav_people": EnvSpec(
        "3. warehouse navigation, IRA people walking", ira=True, num_humans=6,
        build_props=_nav("build_nav_people"),
    ),
    "nav_people_boxes": EnvSpec(
        "4. env 3 + big and small box in front of the robot", ira=True, num_humans=6,
        build_props=_nav("build_nav_people_boxes"),
    ),
    "nav_people_forearm_box": EnvSpec(
        "5. env 3 + box attached to the robot's forearms", ira=True, num_humans=6,
        build_props=_nav("build_nav_people"), post_robot=_nav("attach_forearm_box"),
    ),
    "social_static": EnvSpec(
        "6. warehouse + 4 named static humans (LLM social-nav eval, /sim/humans)",
        build_props=_social("build_social_static"), post_robot=_social("attach_humans_publisher"),
        static_targets=False,
    ),
}


def get(name: str) -> EnvSpec:
    if name not in ENVIRONMENTS:
        raise KeyError(f"unknown --env {name!r}; choose from {sorted(ENVIRONMENTS)}")
    return ENVIRONMENTS[name]
