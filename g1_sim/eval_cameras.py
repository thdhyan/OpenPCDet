"""Evaluation cameras for ``--record-dir``, 1920x1080 JPEG sequences
``<record-dir>/<camera>/<step>.jpg``:

* ``robot_pov``   - the G1's own head D435 (what the robot sees),
* ``<Name>_pov``  - each human under ``/World/Humans`` (env 6), eye height,
                    looking along their facing (what that person sees),
* ``top``         - global top-down view over the robot and all humans,
* ``follow``      - chase camera behind the G1.

Render products only render on capture steps (they are parked in between),
so the extra 1080p views cost little sim speed. Encode with
``scripts/frames_to_mp4.sh <record-dir>``.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CAM_ROOT = "/World/EvalCams"
FOLLOW_BACK = 2.6  # m behind the pelvis
FOLLOW_UP = 1.9    # m above the ground
EYE_HEIGHT = 1.6   # m, human POV
TOP_HEIGHT = 8.5   # m, top-down view - below the baked warehouse ceiling (9.0 m)
RESOLUTION = (1920, 1080)


def _look_at(stage, path: str, eye, target, focal_mm: float, up=(0.0, 0.0, 1.0)) -> None:
    from pxr import Gf, UsdGeom

    cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
    if not cam:
        cam = UsdGeom.Camera.Define(stage, path)
        cam.CreateFocalLengthAttr(focal_mm)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 300.0))
        UsdGeom.Xformable(cam).AddTransformOp()
    eye, target = [float(v) for v in eye], [float(v) for v in target]  # Gf rejects numpy scalars
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    UsdGeom.Xformable(cam).GetOrderedXformOps()[0].Set(view.GetInverse())


class EvalRecorder:
    def __init__(self, stage, out_dir: Path, humans: list[dict], robot_camera: str | None = None,
                 every: int = 12, res=RESOLUTION):
        import omni.replicator.core as rep

        if every < 4:
            raise ValueError("--record-every must be >= 4 (views render for 2 frames before each capture)")

        self.stage, self.out_dir, self.every = stage, Path(out_dir), every
        cams = {}
        if robot_camera:
            cams["robot_pov"] = robot_camera
        for h in humans:
            a = math.radians(h["heading_deg"])
            fwd = (math.cos(a), math.sin(a))
            eye = (h["x"] + 0.15 * fwd[0], h["y"] + 0.15 * fwd[1], EYE_HEIGHT)  # just in front of the face
            path = f"{CAM_ROOT}/{h['name']}_pov"
            # gaze slightly down, so a robot walking up to the person stays in frame
            _look_at(stage, path, eye, (eye[0] + 4 * fwd[0], eye[1] + 4 * fwd[1], 0.8), 14.0)
            cams[f"{h['name']}_pov"] = path
        xs = [0.0] + [h["x"] for h in humans]
        ys = [0.0] + [h["y"] for h in humans]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        _look_at(stage, f"{CAM_ROOT}/top", (cx, cy, TOP_HEIGHT), (cx, cy, 0.0), 18.0, up=(0.0, 1.0, 0.0))
        cams["top"] = f"{CAM_ROOT}/top"
        _look_at(stage, f"{CAM_ROOT}/follow", (0.0, -FOLLOW_BACK, FOLLOW_UP), (0.0, 0.5, 0.8), 16.0)
        cams["follow"] = f"{CAM_ROOT}/follow"

        self.views = {}
        for name, path in cams.items():
            rp = rep.create.render_product(path, list(res))
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach(rp)
            rp.hydra_texture.set_updates_enabled(False)
            self.views[name] = (rp, ann)
            (self.out_dir / name).mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=4)  # JPEG encoding off the sim thread
        print(f"[WH] eval cameras    : {', '.join(cams)} {res[0]}x{res[1]} -> {self.out_dir} every {every} steps")

    def step(self, step: int, pelvis_pos, pelvis_quat_wxyz) -> None:
        """Re-aim the chase cam; render the views on the step before a capture, save on the capture step."""
        from PIL import Image

        p = np.asarray(pelvis_pos).reshape(-1)[:3]
        w, x, y, z = np.asarray(pelvis_quat_wxyz).reshape(-1)[:4]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        fwd = (math.cos(yaw), math.sin(yaw))
        eye = (p[0] - FOLLOW_BACK * fwd[0], p[1] - FOLLOW_BACK * fwd[1], FOLLOW_UP)
        _look_at(self.stage, f"{CAM_ROOT}/follow", eye, (p[0] + 0.8 * fwd[0], p[1] + 0.8 * fwd[1], 0.8), 16.0)

        phase = step % self.every
        if phase == self.every - 2:  # views render on the next two steps (RTX settles, no stale frame)
            for rp, _ in self.views.values():
                rp.hydra_texture.set_updates_enabled(True)
        elif phase == 0:
            for name, (rp, ann) in self.views.items():
                img = np.asarray(ann.get_data())
                rp.hydra_texture.set_updates_enabled(False)
                if img.ndim == 3 and img.size:
                    frame = Image.fromarray(img[..., :3].astype(np.uint8))
                    self.pool.submit(frame.save, self.out_dir / name / f"{step:07d}.jpg", quality=90)
