"""Warehouse environment loading for the SDG pipeline.

Fixes a real bug found in ``g1_rtx_sim.py``'s ``_build_room()``: it passed a
bare Nucleus-catalog-relative path (``/Isaac/Environments/Simple_Room/...``)
straight to ``isaacsim.core.utils.stage.add_reference_to_stage``, which does
**not** resolve catalog-relative paths against the install's asset root - it
needs either a fully qualified URL or a local file path. The prior code's
``try/except`` silently caught the resulting failure and fell back to flat
ground on every run, which is why the room "never loaded" despite the asset
genuinely existing (curl-verified HTTP 200 against the CDN root earlier this
session). The fix is the same one-liner used everywhere else in Isaac Sim
sample code: resolve through ``get_assets_root_path()`` first.
"""

from __future__ import annotations

import os

WAREHOUSE_USD = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
WAREHOUSE_FORKLIFTS_USD = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
SIMPLE_ROOM_USD = "/Isaac/Environments/Simple_Room/simple_room.usd"


def resolve_nucleus_path(rel_path: str) -> str:
    """Resolve a Nucleus-catalog-relative path (e.g. ``/Isaac/Environments/...``)
    against this install's real asset root.

    Raises if the asset root cannot be determined - callers should catch and
    fall back rather than silently passing an unresolved path downstream
    (that silent failure is exactly the bug this function exists to fix).
    """
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    if not root:
        raise RuntimeError("get_assets_root_path() returned nothing - no asset root configured")
    return root + rel_path


def load_environment(stage, usd_rel_path: str = WAREHOUSE_USD, prim_path: str = "/World/Env") -> str:
    """Reference a Nucleus or local environment USD onto the stage.

    ``/Isaac/...`` paths are resolved through the current Nucleus asset root;
    all other paths are treated as local USD paths. Returns the resolved path
    actually used (for logging). Raises on failure so callers can choose the
    flat-ground fallback.
    """
    import isaacsim.core.utils.stage as stage_utils

    if usd_rel_path.startswith("/Isaac/"):
        resolved = resolve_nucleus_path(usd_rel_path)
    else:
        local_path = os.path.expanduser(usd_rel_path)
        if not os.path.isabs(local_path):
            local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), local_path)
        resolved = os.path.abspath(local_path)
    stage_utils.add_reference_to_stage(usd_path=resolved, prim_path=prim_path)
    return resolved


def build_flat_ground(stage, prim_path: str = "/World/ground") -> None:
    """The pre-existing flat-ground fallback, factored out so both the
    warehouse and simple_room callers share one implementation."""
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    ground = UsdGeom.Cube.Define(stage, prim_path)
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.1))
    ground.AddScaleOp().Set(Gf.Vec3f(120.0, 120.0, 0.2))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    light = UsdLux.DistantLight.Define(stage, "/World/light")
    light.CreateIntensityAttr(3000.0)
