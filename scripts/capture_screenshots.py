#!/usr/bin/env python3
"""Screenshot pass for the G1 warehouse sim (Isaac Sim 6.1.0).

Captures, in ONE headless run of the proven g1_warehouse_sim scene:

* ``front_camera_rgb.png`` / ``front_camera_depth.png`` - the D435 on the G1
  torso (the same camera prim the /g1/camera/* ROS topics render from)
* ``env_front.png`` / ``env_side.png`` / ``env_overview.png`` - the warehouse
  from 3 free camera positions (Replicator render products, 1920x1080 default)
* ``lidar_debug_viewport.png`` - viewport screenshot with the MID360 points
  "turned on", two ways stacked:
    - official NVIDIA path: ``RtxSensorDebugDrawPointCloud`` writer
      (``draw-point-cloud`` from ``isaacsim.sensors.rtx.nodes``) attached to a
      1x1 render product on the lidar prim
    - ``omni.debugdraw`` elevation-colored accumulation of the last ~2 sweeps
      (same mechanism scripts/lidar_viz_isaacsim.py already proved on this box)
* ``lidar_overlay_<view>.png`` - that accumulated cloud projected into the
  env/D435 images with cv2 (guaranteed output even if viewport debug draw is
  invisible headless): the classic "points drawn over the camera image" look.

The scene boots via ``import g1_warehouse_sim`` so the charset_normalizer
pre-import pin, sensor carb settings and experience selection stay identical
to the proven entry point; its CLI is forced to ``--headless --no-ira
--no-ros2`` (WBC Balance still runs internally - cmd_vel simply stays 0).

Usage (from G1_sim/, uv venv active):
    python scripts/capture_screenshots.py --out screenshots/demo
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

# --- our flags FIRST: g1_warehouse_sim parses sys.argv at import time -------
ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("--out", type=str, default=None, help="Output dir (default screenshots/<timestamp>)")
ap.add_argument("--warmup", type=int, default=240, help="Settle steps before capturing (WBC standing + lidar sweeps)")
ap.add_argument("--env-res", type=str, default="1920x1080", help="WxH for the environment view captures")
ap.add_argument(
    "--sweep-steps",
    type=int,
    default=12,
    help="Rolling lidar accumulation window in sim steps (~6 steps = one 10 Hz full sweep)",
)
ap.add_argument(
    "--dump-pcd",
    action="store_true",
    help="Also save the final cloud to <out>/lidar_cloud.pcd (ASCII; intensity = elevation deg, same ramp as the overlays)",
)
cli = ap.parse_args()

ENV_W, ENV_H = (int(v) for v in cli.env_res.lower().split("x"))

# Force g1_warehouse_sim's CLI onto the proven capture config BEFORE import:
# its charset pin + SimulationApp boot then run at import, original order.
sys.argv = ["g1_warehouse_sim.py", "--headless", "--no-ira", "--no-ros2"]
import g1_warehouse_sim as gw  # noqa: E402  (boots SimulationApp at module level)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

REPO = gw.REPO
# Absolute from the start: save_png logs paths relative to REPO, and a bare
# relative --out broke that (ValueError in pathlib.relative_to, run 2026-09-22).
OUT_DIR = (
    Path(cli.out).expanduser().resolve()
    if cli.out
    else REPO / "screenshots" / time.strftime("%Y%m%d_%H%M%S")
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Free camera positions for the 3 environment views. Poses are initial
# guesses around the Simple_Warehouse floor (robot at origin, pedestrian
# target boxes at x=3..6 y=-2..2.5); the front view is offset +y so the
# line of sight clears the target box sitting at (3, 0).
ENV_VIEWS = [
    # name,        eye (x,y,z),            look-at (x,y,z),          focal mm
    ("front", (9.0, 1.4, 1.8), (0.0, 0.0, 1.05), 20.0),  # looking back at the G1 past the target boxes; y=3.2 was inside a rack (black frame), y=0.5 occluded by the (3,0) box
    ("side", (-4.0, -7.0, 1.6), (3.0, 0.5, 0.9), 24.0),  # diagonal across the aisle
    ("overview", (7.0, -8.0, 5.5), (1.0, 0.0, 0.4), 16.0),  # elevated wide shot
]


def log(msg: str) -> None:
    print(f"[CAP] {msg}", flush=True)


# ---------------------------------------------------------------------------
# scene bootstrap (mirrors g1_warehouse_sim.main(), minus ROS2 / IRA)
# ---------------------------------------------------------------------------
gw.enable_extensions()
mgr = omni.kit.app.get_app().get_extension_manager()
# Registers the "draw-point-cloud" / RtxSensorDebugDrawPointCloud writer -
# must precede rep.writers.get() below (official RTX annotator docs).
mgr.set_extension_enabled_immediate("isaacsim.sensors.rtx.nodes", True)
for _ in range(10):
    omni.kit.app.get_app().update()
gw._log_sensor_settings()

from isaacsim.core.api import SimulationContext  # noqa: E402  (ext enabled above)

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

gw.build_scene_fallback(stage)  # warehouse reference + 3 pedestrian target boxes
gw.spawn_g1(stage)
for _ in range(10):
    gw.simulation_app.update()
pelvis = stage.GetPrimAtPath(f"{gw.ROBOT_PRIM}/pelvis")
log(f"G1 prim valid={stage.GetPrimAtPath(gw.ROBOT_PRIM).IsValid()} pelvis={pelvis.IsValid()}")

sim = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / gw.SIM_RATE_HZ,
    rendering_dt=1.0 / gw.SIM_RATE_HZ,
)

from isaacsim.core.prims import Articulation  # noqa: E402

robot = Articulation(f"{gw.ROBOT_PRIM}/pelvis", name="g1")
sim.reset()
log(f"articulation {robot.num_dof} DOF")

# WBC standing pose: copy of g1_warehouse_sim's gains + bridge block (its
# loop body keeps cmd_vel at 0 without ROS2 -> Balance policy stands still).
from g1_sim.wbc_bridge import (  # noqa: E402
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

dof_names = set(robot.dof_names or [])
missing = [j for j in ALL_JOINTS if j not in dof_names]
if missing:
    raise SystemExit(f"[CAP] USD is missing joints {missing} - cannot pose the robot")
robot.set_gains(kps=KP[None, :], kds=KD[None, :], joint_names=LEG_WAIST_JOINTS)
robot.set_gains(
    kps=np.full((1, len(ARM_JOINTS)), ARM_KP, dtype=np.float32),
    kds=np.full((1, len(ARM_JOINTS)), ARM_KD, dtype=np.float32),
    joint_names=ARM_JOINTS,
)
wbc = WbcBridge(
    REPO / "assets/policy/GR00T-WholeBodyControl-Balance.onnx",
    REPO / "assets/policy/GR00T-WholeBodyControl-Walk.onnx",
)
log("WBC bridge loaded (Balance/Walk, gains overridden)")

# Sensors: D435 on the torso, MID360 on mid360_link (identical args to main).
camera_prim = gw.spawn_camera(f"{gw.ROBOT_PRIM}/torso_link")
mount = f"{gw.ROBOT_PRIM}/mid360_link"
if not stage.GetPrimAtPath(mount).IsValid():
    raise SystemExit(f"[CAP] mount prim {mount} missing")
prim_paths = gw.spawn_mid360(
    mount,
    config_dir=REPO / "assets/lidar_configs_rotary",
    translation=(0.0, 0.0, -0.03),  # matches g1_warehouse_sim (GMO fix 2026-09-22)
    orientation=gw.MID360_QUAT_WXYZ,
)
log(f"lidar prims: {prim_paths}")
sp = stage.GetPrimAtPath(prim_paths[0])
if sp.IsValid():
    wp = UsdGeom.Xformable(sp).ComputeLocalToWorldTransform(0.0).ExtractTranslation()
    log(f"sensor world pos: ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})")

# Free environment cameras (USD only for now - render products are created
# one at a time during capture to keep 8 GB VRAM headroom).
def define_view_camera(path: str, eye, target, focal_mm: float) -> str:
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal_mm)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 300.0))
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0)
    )
    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(view.GetInverse())
    return path


env_cam_paths = {}
for name, eye, tgt, focal in ENV_VIEWS:
    env_cam_paths[name] = define_view_camera(f"/World/Capture/{name}", eye, tgt, focal)
log(f"env cameras: {list(env_cam_paths)}")

# Official debug-draw writer: tiny render product on the first lidar prim.
lidar_writer = None
lidar_rp = rep.create.render_product(prim_paths[0], [1, 1])
try:
    lidar_writer = rep.writers.get("RtxSensorDebugDrawPointCloud")
    if lidar_writer is None:
        log("WARNING: RtxSensorDebugDrawPointCloud writer not registered")
    else:
        lidar_writer.initialize(size=0.08, color=[1.0, 0.35, 0.05, 1.0])
        lidar_writer.attach([lidar_rp])
        log("draw-point-cloud writer attached (official debug viz)")
except Exception as e:  # noqa: BLE001
    log(f"WARNING: could not attach debug writer: {e}")
    lidar_writer = None

# LidarSensor wrappers are REQUIRED to enable GMO annotators (rtx_publisher
# docstring); same proven pattern, minus rclpy.
from isaacsim.sensors.experimental.rtx import (  # noqa: E402
    LidarSensor,
    parse_generic_model_output_data,
)

lidar_sensors = [LidarSensor(p, annotators=["generic-model-output"]) for p in prim_paths]
# Rolling window ~2 sweeps of sensor-local points per prim.
sweeps: list[deque] = [deque(maxlen=cli.sweep_steps) for _ in prim_paths]


def gather(sensor) -> np.ndarray | None:
    """One render's Cartesian returns (sensor-local), filtered like rtx_publisher."""
    data, _info = sensor.get_data("generic-model-output")
    if data is None:
        return None
    gmo = parse_generic_model_output_data(data)
    if gmo.x is None or len(gmo.x) == 0:
        return None
    coords = getattr(gmo, "elementsCoordsType", None)
    cname = getattr(coords, "name", str(coords)) if coords is not None else "CARTESIAN"
    if "SPHERICAL" in cname.upper():
        az = np.radians(np.asarray(gmo.x, dtype=np.float32))
        el = np.radians(np.asarray(gmo.y, dtype=np.float32))
        r = np.asarray(gmo.z, dtype=np.float32)
        xyz = np.stack(
            [r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)],
            axis=1,
        )
    else:
        xyz = np.stack(
            [np.asarray(gmo.x), np.asarray(gmo.y), np.asarray(gmo.z)], axis=1
        ).astype(np.float32)
    r = np.linalg.norm(xyz, axis=1)
    keep = np.isfinite(xyz).all(axis=1) & (r > 1e-6) & (r <= 40.0)
    xyz = xyz[keep]
    return xyz if len(xyz) else None


def accumulated_world_points() -> np.ndarray | None:
    """Rolling sweeps transformed sensor-local -> world."""
    chunks = [np.vstack(d) for d in sweeps if d]
    if not chunks:
        return None
    local = np.vstack(chunks)
    m = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(prim_paths[0]))
    M = np.array([list(m.GetRow(i)) for i in range(4)])  # row-vector convention
    h = np.hstack([local, np.ones((len(local), 1), dtype=np.float64)])
    return (h @ M)[:, :3]


# ---------------------------------------------------------------------------
# warmup: WBC standing + lidar accumulation
# ---------------------------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()
timeline.play()
log(f"timeline playing={timeline.is_playing()} warmup={cli.warmup} steps")

wbc_period = 1.0 / gw.WBC_CONTROL_HZ
wbc_next = 0.0
wbc_updates = 0
t0 = time.time()
try:
    for step in range(1, cli.warmup + 1):
        sim.step(render=True)

        if sim.current_time >= wbc_next:
            wbc_next = sim.current_time + wbc_period
            qpos = np.asarray(robot.get_joint_positions(joint_names=ALL_JOINTS))[0]
            qvel = np.asarray(robot.get_joint_velocities(joint_names=ALL_JOINTS))[0]
            _, quat_wxyz = robot.get_world_poses()
            quat_wxyz = np.asarray(quat_wxyz)[0]
            ang_world = np.asarray(robot.get_angular_velocities())[0]
            ang_body = quat_rotate_inverse(quat_wxyz, ang_world)
            target = wbc.step(qpos, qvel, quat_wxyz, ang_body, 0.0, 0.0, 0.0)
            robot.set_joint_position_targets(target[None, :].astype(np.float32), joint_names=LEG_WAIST_JOINTS)
            if wbc_updates == 0:
                arm_pose = np.zeros((1, len(ARM_JOINTS)), dtype=np.float32)
                arm_pose[0, 3] = -1.2   # left_elbow_joint
                arm_pose[0, 10] = -1.2  # right_elbow_joint
                robot.set_joint_position_targets(arm_pose, joint_names=ARM_JOINTS)
            wbc_updates += 1

        for i, s in enumerate(lidar_sensors):
            pts = gather(s)
            if pts is not None:
                sweeps[i].append(pts)

        if step % 60 == 0:
            pz = robot.get_world_poses()[0][0][2]
            n = sum(len(d) for d in sweeps)
            log(f"step {step:>4}  pelvis_z={pz:.3f}  acc_pts={n}  wbc_updates={wbc_updates}")
finally:
    pelvis_z = float(robot.get_world_poses()[0][0][2])
log(f"warmup done in {time.time() - t0:.1f}s  pelvis_z={pelvis_z:.3f}")
if pelvis_z < 0.5:
    log("WARNING: robot appears collapsed (pelvis_z < 0.5) - screenshots will show it on the floor")

world_pts = accumulated_world_points()
if world_pts is None or len(world_pts) == 0:
    log("WARNING: no lidar points accumulated - debug shots will be empty")
    world_pts = np.zeros((0, 3))
else:
    # stride keeps debug-draw frame cost sane while preserving the pattern
    stride = max(1, len(world_pts) // 30000)
    world_pts = world_pts[::stride]
    log(f"accumulated lidar: {len(world_pts)} world points (stride {stride})")

# sensor origin in world (for ray fan lines)
sensor_origin = UsdGeom.XformCache().GetLocalToWorldTransform(
    stage.GetPrimAtPath(prim_paths[0])
).ExtractTranslation()


# ---------------------------------------------------------------------------
# capture helpers
# ---------------------------------------------------------------------------
def rgba_to_rgb(data) -> np.ndarray | None:
    if isinstance(data, dict):
        for k in ("rgba", "Rgba", "rgb", "RGB"):
            if k in data and np.asarray(data[k]).size:
                data = data[k]
                break
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.size == 0:
        return None
    if arr.shape[-1] == 4:
        return arr[..., :3].copy()
    if arr.shape[-1] == 3:
        return arr.copy()
    return None


def capture_cam_rgb(cam_path: str, res: tuple[int, int], with_lidar: bool = False) -> np.ndarray | None:
    """Transient render product + rgb annotator on a camera prim.

    ``with_lidar=True`` redraws the debug-draw cloud every frame during the
    capture so the shot consistently contains the lidar points (the official
    draw-point-cloud writer only fires on its own schedule and showed up in
    just one of three captures).
    """
    rp = rep.create.render_product(cam_path, list(res))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach(rp)
    img = None
    try:
        for _ in range(15):
            if with_lidar:
                draw_lidar()
            sim.step(render=True)
            img = rgba_to_rgb(ann.get_data())
            if img is not None and not with_lidar:
                break
            if img is not None and with_lidar and _ >= 4:
                # take a couple of extra frames so at least one fresh draw
                # is guaranteed inside the captured buffer
                break
    finally:
        try:
            ann.detach(rp)
            rp.destroy()
        except Exception as e:  # noqa: BLE001
            log(f"render product teardown: {e}")
    return img


def cam_projection(cam_path: str):
    """(view_matrix_np, fx, fy, cx, cy) for a UsdGeom.Camera prim."""
    prim = stage.GetPrimAtPath(cam_path)
    cam = UsdGeom.Camera(prim)
    M = UsdGeom.XformCache().GetLocalToWorldTransform(prim).GetInverse()
    view = np.array([list(M.GetRow(i)) for i in range(4)])
    f = cam.GetFocalLengthAttr().Get()
    hap = cam.GetHorizontalApertureAttr().Get()
    vap = cam.GetVerticalApertureAttr().Get()
    return view, f, hap, vap


def overlay_points(
    img_rgb: np.ndarray, cam_path: str, pts: np.ndarray, radius: int = 2
) -> np.ndarray:
    """Draw the world point cloud into a camera image, colored by elevation.

    USD camera looks down -Z, image u right / v down:
        u = cx + fx * Xc / -Zc ,  v = cy - fy * Yc / -Zc
    """
    import cv2

    if len(pts) == 0:
        return img_rgb
    view, f, hap, vap = cam_projection(cam_path)
    H, W = img_rgb.shape[:2]
    fx = W * f / hap
    fy = H * f / vap
    h = np.hstack([pts.astype(np.float64), np.ones((len(pts), 1))])
    cam = h @ view
    xc, yc, zc = cam[:, 0], cam[:, 1], cam[:, 2]
    front = zc < -0.05
    if not np.any(front):
        return img_rgb
    xc, yc, zc = xc[front], yc[front], zc[front]
    u = (W / 2.0 + fx * xc / -zc).round().astype(np.int32)
    v = (H / 2.0 - fy * yc / -zc).round().astype(np.int32)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[inb], v[inb]
    if len(u) == 0:
        return img_rgb

    # elevation relative to the sensor origin -> turbo colormap (blue low,
    # red high), same spirit as lidar_viz_isaacsim._el_to_color
    sub = pts[front][inb]
    rel = sub - np.asarray(sensor_origin)
    rng = np.linalg.norm(rel, axis=1) + 1e-9
    el = np.degrees(np.arcsin(np.clip(rel[:, 2] / rng, -1, 1)))
    el_u8 = np.clip((el + 7.2) / 59.4, 0, 1)  # profile band -7.2..+52.2
    colors = cv2.applyColorMap((el_u8 * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    colors = colors.reshape(-1, 3)  # BGR

    out = img_rgb.copy()
    bgr = out[..., ::-1].copy()  # cv2 works in BGR
    for (ui, vi), col in zip(zip(u, v), colors):
        cv2.circle(bgr, (int(ui), int(vi)), radius, tuple(int(c) for c in col), -1)
    return bgr[..., ::-1].copy()


def save_png(path: Path, img_rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(img_rgb.astype(np.uint8)).save(path)
    try:
        shown = path.relative_to(REPO)
    except ValueError:
        shown = path
    log(f"saved {shown}  {img_rgb.shape[1]}x{img_rgb.shape[0]}")


def save_depth(path: Path, depth: np.ndarray, cap_m: float = 10.0) -> None:
    import cv2

    d = np.where(np.isfinite(depth), depth, 0.0)
    d = np.clip(d, 0.0, cap_m) / cap_m * 255.0
    cm = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), cm)
    log(f"saved {path.relative_to(REPO)}  (depth, 0-{cap_m:.0f} m, TURBO)")


def capture_depth(cam_path: str, res: tuple[int, int]) -> np.ndarray | None:
    rp = rep.create.render_product(cam_path, list(res))
    ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    ann.attach(rp)
    depth = None
    try:
        for _ in range(15):
            sim.step(render=True)
            data = ann.get_data()
            if isinstance(data, dict):
                data = data.get("data", next(iter(data.values()), None))
            arr = np.asarray(data) if data is not None else None
            if arr is not None and arr.size:
                arr = arr.astype(np.float32)
                depth = arr.reshape(res[1], res[0]) if arr.ndim == 1 else arr
                break
    finally:
        try:
            ann.detach(rp)
            rp.destroy()
        except Exception as e:  # noqa: BLE001
            log(f"depth RP teardown: {e}")
    return depth


def el_colors_abgr(pts_rel: np.ndarray) -> list[int]:
    rng = np.linalg.norm(pts_rel, axis=1) + 1e-9
    el = np.degrees(np.arcsin(np.clip(pts_rel[:, 2] / rng, -1, 1)))
    t = np.clip((el + 7.2) / 59.4, 0, 1)
    r = (t * 255).astype(np.int32)
    b = ((1 - t) * 255).astype(np.int32)
    # 0xAARRGGBB for omni.debugdraw (run 2026-09-22: packing BB/RR rendered
    # every point red; np.int32 | int also overflows under numpy 2, hence int()).
    return [
        int((0xFF << 24) | (int(rr) << 16) | int(bb))
        for bb, rr in zip(b, r)
    ]


# ---------------------------------------------------------------------------
# debug-draw cloud setup (must precede the captures: capture_cam_rgb can
# redraw it every frame via draw_lidar)
# ---------------------------------------------------------------------------
debug_draw = None
have_lines = False
try:
    from omni.debugdraw import get_debug_draw_interface

    debug_draw = get_debug_draw_interface()
except Exception as e:  # noqa: BLE001
    log(f"WARNING: omni.debugdraw unavailable: {e}")

try:
    import carb
except Exception:  # noqa: BLE001
    carb = None

rel_pts = world_pts - np.asarray(sensor_origin)
abgr = el_colors_abgr(rel_pts) if len(world_pts) else []


def draw_lidar() -> None:
    global have_lines, debug_draw
    if debug_draw is None or carb is None or not len(world_pts):
        return
    try:
        for pt, col in zip(world_pts, abgr):
            debug_draw.draw_point(
                carb.Float3(float(pt[0]), float(pt[1]), float(pt[2])), col, 3.0
            )
    except Exception as e:  # noqa: BLE001 - never let drawing kill the capture
        log(f"WARNING: debugdraw draw_point failed, disabling: {e!r}")
        debug_draw = None
        return
    if have_lines or "draw_line" not in dir(debug_draw):
        return
    try:  # one probe first: ray fan only where draw_line exists
        debug_draw.draw_line(
            carb.Float3(float(sensor_origin[0]), float(sensor_origin[1]), float(sensor_origin[2])),
            carb.Float3(float(world_pts[0][0]), float(world_pts[0][1]), float(world_pts[0][2])),
            0x60FFFFFF,
        )
        have_lines = True
    except Exception:  # noqa: BLE001
        have_lines = False
        log("draw_line not supported - points only")
    if have_lines:
        ray_stride = max(1, len(world_pts) // 3000)
        for pt in world_pts[::ray_stride]:
            debug_draw.draw_line(
                carb.Float3(float(sensor_origin[0]), float(sensor_origin[1]), float(sensor_origin[2])),
                carb.Float3(float(pt[0]), float(pt[1]), float(pt[2])),
                0x60FFFFFF,
            )


# ---------------------------------------------------------------------------
# 1) D435 front camera (RGB + false-color depth)
# ---------------------------------------------------------------------------
d435_rgb = None
try:
    log("capturing D435 front camera...")
    d435_rgb = capture_cam_rgb(camera_prim, (640, 480))
    if d435_rgb is not None:
        save_png(OUT_DIR / "front_camera_rgb.png", d435_rgb)
    else:
        log("WARNING: D435 rgb annotator returned nothing")
    depth = capture_depth(camera_prim, (640, 480))
    if depth is not None:
        save_depth(OUT_DIR / "front_camera_depth.png", depth)
    else:
        log("WARNING: D435 depth annotator returned nothing")
except Exception as e:  # noqa: BLE001 - keep going: later shots matter too
    log(f"WARNING: D435 capture failed: {e!r}")

# ---------------------------------------------------------------------------
# 2) three environment views (one transient render product at a time)
# ---------------------------------------------------------------------------
env_imgs: dict[str, np.ndarray] = {}
for name, _, _, _ in ENV_VIEWS:
    try:
        log(f"capturing env view '{name}'...")
        img = capture_cam_rgb(env_cam_paths[name], (ENV_W, ENV_H))
        if img is not None:
            env_imgs[name] = img
            save_png(OUT_DIR / f"env_{name}.png", img)
        else:
            log(f"WARNING: env view '{name}' returned nothing")
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: env view '{name}' failed: {e!r}")

# lidar-pattern variants: debug cloud redrawn every frame during capture
for name, _, _, _ in ENV_VIEWS:
    if name not in env_imgs:
        continue
    try:
        log(f"capturing env view '{name}' WITH lidar debug points...")
        img = capture_cam_rgb(env_cam_paths[name], (ENV_W, ENV_H), with_lidar=True)
        if img is not None:
            save_png(OUT_DIR / f"env_{name}_lidar.png", img)
        else:
            log(f"WARNING: env lidar view '{name}' returned nothing")
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: env lidar view '{name}' failed: {e!r}")

# ---------------------------------------------------------------------------
# 3) lidar debug viz: viewport screenshot (official writer + debugdraw cloud)
# ---------------------------------------------------------------------------
log("viewport capture with lidar debug points...")
viewport_file = OUT_DIR / "lidar_debug_viewport.png"
vp_log = "skipped"
try:
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    vp = get_active_viewport()
    try:
        vp_log = f"{vp.width}x{vp.height}"
    except Exception:  # noqa: BLE001
        vp_log = "?"
    # point the viewport at the overview camera (fallback: manipulate its own)
    try:
        vp.camera_path = env_cam_paths["overview"]
    except Exception:  # noqa: BLE001
        try:
            vp.set_camera(
                Gf.Vec3d(*ENV_VIEWS[2][1]), Gf.Vec3d(*ENV_VIEWS[2][2])
            )
        except Exception as e:  # noqa: BLE001
            log(f"viewport camera switch failed: {e}")
    # Prime: run 2026-09-22 captured the frame right after the camera switch
    # (black bars, stale composite). Draw + render a few frames FIRST, then
    # trigger the capture on a settled viewport.
    for _ in range(8):
        draw_lidar()
        sim.step(render=True)
    capture_viewport_to_file(vp, str(viewport_file))
    for _ in range(40):  # pump frames: draw each frame + let the capture land
        draw_lidar()
        sim.step(render=True)
        if viewport_file.exists() and viewport_file.stat().st_size > 0:
            break
    log(f"viewport {vp_log} -> {viewport_file.name} exists={viewport_file.exists()}")
except Exception as e:  # noqa: BLE001
    log(f"WARNING: viewport capture failed: {e}")

# keep the debug draw alive for a few more frames (in case of slow capture)
for _ in range(10):
    draw_lidar()
    sim.step(render=True)

# ---------------------------------------------------------------------------
# 4) guaranteed output: projected overlay onto env + D435 images
# ---------------------------------------------------------------------------
log("projecting lidar overlay onto camera images...")
targets: dict[str, tuple[str, np.ndarray]] = {
    f"lidar_overlay_{name}.png": (env_cam_paths[name], img) for name, img in env_imgs.items()
}
if d435_rgb is not None:
    targets["lidar_overlay_d435.png"] = (camera_prim, d435_rgb)
for fname, (cam_path, img) in targets.items():
    try:
        out = overlay_points(img, cam_path, world_pts, radius=2)
        save_png(OUT_DIR / fname, out)
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: overlay {fname} failed: {e!r}")

# ---------------------------------------------------------------------------
manifest = [
    f"front_camera_rgb.png      - D435 RGB (640x480, torso mount, pitched down ~47.6 deg)",
    f"front_camera_depth.png    - D435 depth, TURBO false color, 0-10 m",
    *[f"env_{name:<18} - free camera {eye} -> {tgt}, {ENV_W}x{ENV_H}" for name, eye, tgt, _ in ENV_VIEWS],
    "lidar_debug_viewport.png  - viewport (overview cam) + RtxSensorDebugDrawPointCloud writer",
    "                            + omni.debugdraw elevation-colored accumulation (blue low -> red high)",
    *[f"{f:<25} - lidar cloud projected into that image (cv2, elevation colored)" for f in targets],
    "",
    f"warmup={cli.warmup} steps  pelvis_z={pelvis_z:.3f}  drawn_pts={len(world_pts)}  "
    f"rays={'yes' if have_lines else 'no'}  writer={'yes' if lidar_writer else 'no'}",
]
(OUT_DIR / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")

# Optional PCD export of the final accumulated cloud (world frame).
if cli.dump_pcd:
    if len(world_pts):
        pcd_path = OUT_DIR / "lidar_cloud.pcd"
        rel = world_pts - np.asarray(sensor_origin)
        rng = np.linalg.norm(rel, axis=1) + 1e-9
        el = np.degrees(np.arcsin(np.clip(rel[:, 2] / rng, -1, 1)))
        with open(pcd_path, "w") as fh:
            fh.write("# .PCD v0.7 - Point Cloud Data file format\n")
            fh.write("VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n")
            fh.write("COUNT 1 1 1 1\n")
            fh.write(f"WIDTH {len(world_pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
            fh.write(f"POINTS {len(world_pts)}\nDATA ascii\n")
            np.savetxt(fh, np.column_stack([world_pts, el]), fmt="%.4f %.4f %.4f %.2f")
        log(f"dumped {pcd_path.name}  {len(world_pts)} pts (intensity = elevation deg)")
    else:
        log("dump-pcd: no points to dump")

log("---- done ----")
for p in sorted(OUT_DIR.iterdir()):
    log(f"  {p.name}  ({p.stat().st_size} bytes)")

timeline.stop()
gw.simulation_app.close()
