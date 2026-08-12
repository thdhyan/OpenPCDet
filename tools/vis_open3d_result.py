#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize detection results (.npz with points/pred_boxes/pred_scores/pred_labels/class_names)
from infer_headless.py (OpenPCDet) or livox_detection's infer_csv.py, with Open3D.

Usage:
    python vis_open3d_result.py --npz path/to/result.npz [--score_thresh 0.0] [--save_png out.png]
"""

import argparse
import numpy as np
import open3d as o3d

DEFAULT_COLOR_MAP = {
    'Car': [0, 1, 1], 'Vehicle': [0, 1, 1],
    'Pedestrian': [1, 1, 0],
    'Cyclist': [0, 1, 0],
}

LINES = [[0, 1], [1, 2], [2, 3], [3, 0],
         [4, 5], [5, 6], [6, 7], [7, 4],
         [0, 4], [1, 5], [2, 6], [3, 7]]


def boxes_to_corners_3d(boxes3d):
    """boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading] -> (N, 8, 3) corners."""
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ], dtype=np.float32) / 2

    dims = boxes3d[:, 3:6]
    corners = dims[:, None, :] * template[None, :, :]  # (N, 8, 3)

    angle = boxes3d[:, 6]
    cosa, sina = np.cos(angle), np.sin(angle)
    zeros, ones = np.zeros_like(angle), np.ones_like(angle)
    rot = np.stack([cosa, sina, zeros,
                     -sina, cosa, zeros,
                     zeros, zeros, ones], axis=1).reshape(-1, 3, 3)

    corners = np.einsum('nij,njk->nik', corners, rot)
    corners += boxes3d[:, None, 0:3]
    return corners


def make_box_lineset(corners, color):
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(LINES)
    lineset.colors = o3d.utility.Vector3dVector([color for _ in LINES])
    return lineset


def parse_config():
    parser = argparse.ArgumentParser(description='Visualize detection results')
    parser.add_argument('--npz', type=str, required=True)
    parser.add_argument('--score_thresh', type=float, default=0.0)
    parser.add_argument('--save_png', type=str, default=None, help='save screenshot instead of/in addition to interactive window')
    parser.add_argument('--no_window', action='store_true', help='skip interactive window (for headless screenshot-only)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    data = np.load(args.npz, allow_pickle=True)
    points = data['points']       # (N, 4) x,y,z,intensity
    boxes = data['pred_boxes']    # (M, 7)
    scores = data['pred_scores']  # (M,)
    labels = data['pred_labels']  # (M,) 1-indexed
    class_names = list(data['class_names']) if 'class_names' in data else ['Vehicle', 'Pedestrian', 'Cyclist']

    keep = scores >= args.score_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    print(f'{args.npz}: {points.shape[0]} points, {boxes.shape[0]} boxes (score >= {args.score_thresh})')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    intensity = points[:, 3]
    if intensity.max() > intensity.min():
        norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    else:
        norm = np.zeros_like(intensity)
    pcd.colors = o3d.utility.Vector3dVector(np.column_stack([norm, norm, norm]))

    geometries = [pcd]

    if boxes.shape[0] > 0:
        corners = boxes_to_corners_3d(boxes)
        for i in range(boxes.shape[0]):
            cls_name = class_names[labels[i] - 1]
            color = DEFAULT_COLOR_MAP.get(cls_name, [1, 0, 1])
            geometries.append(make_box_lineset(corners[i], color))
            print(f'  {cls_name:10s} score={scores[i]:.2f}')

    if args.save_png:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=800, visible=False)
        for g in geometries:
            vis.add_geometry(g)
        vis.get_render_option().point_size = 2
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.save_png)
        vis.destroy_window()
        print(f'Saved screenshot to {args.save_png}')

    if not args.no_window:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=args.npz,
            width=1280, height=800,
        )
