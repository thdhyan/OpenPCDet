#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step through per-frame detections (from infer_frames.py) in Open3D.
N / right-arrow = next frame, P / left-arrow = previous frame, Q/Esc = quit.
Prints a running per-class detection count breakdown to the console.

Usage:
    python vis_frames_stepper.py --npz path/to/frames_pointpillar.npz --frames_dir path/to/frames
"""

import argparse
import os
from collections import Counter

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
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ], dtype=np.float32) / 2
    dims = boxes3d[:, 3:6]
    corners = dims[:, None, :] * template[None, :, :]
    angle = boxes3d[:, 6]
    cosa, sina = np.cos(angle), np.sin(angle)
    zeros, ones = np.zeros_like(angle), np.ones_like(angle)
    rot = np.stack([cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones],
                    axis=1).reshape(-1, 3, 3)
    corners = np.einsum('nij,njk->nik', corners, rot)
    corners += boxes3d[:, None, 0:3]
    return corners


def make_box_lineset(corners, color):
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(LINES)
    lineset.colors = o3d.utility.Vector3dVector([color for _ in LINES])
    return lineset


class FrameStepper:
    def __init__(self, npz_path, frames_dir, score_thresh):
        data = np.load(npz_path, allow_pickle=True)
        self.frame_files = list(data['frame_files'])
        self.pred_boxes = data['pred_boxes']
        self.pred_scores = data['pred_scores']
        self.pred_labels = data['pred_labels']
        self.class_names = list(data['class_names'])
        self.frames_dir = frames_dir
        self.score_thresh = score_thresh
        self.idx = 0
        self.n = len(self.frame_files)

        self.total_counts = Counter()
        for labels, scores in zip(self.pred_labels, self.pred_scores):
            keep = scores >= self.score_thresh
            for lab in labels[keep]:
                self.total_counts[self.class_names[lab - 1]] += 1

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name='Frame-wise Detections', width=1280, height=800)
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = 2.0
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.box_geoms = []
        self._first_render = True

        self.vis.register_key_callback(ord('N'), self._next)
        self.vis.register_key_callback(262, self._next)   # right arrow
        self.vis.register_key_callback(ord('P'), self._prev)
        self.vis.register_key_callback(263, self._prev)    # left arrow
        self.vis.register_key_callback(ord('Q'), self._quit)

        self._render_frame()

    def _clear_boxes(self):
        for g in self.box_geoms:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.box_geoms = []

    def _render_frame(self):
        fname = self.frame_files[self.idx]
        pts_path = os.path.join(self.frames_dir, fname)
        points = np.load(pts_path)

        self.pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        intensity = points[:, 3]
        rng = intensity.max() - intensity.min()
        norm = (intensity - intensity.min()) / rng if rng > 0 else np.full_like(intensity, 0.5)
        # blue (low intensity) -> cyan/green (high intensity), visible against dark background
        colors = np.column_stack([norm * 0.3, 0.4 + norm * 0.6, 1.0 - norm * 0.5])
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.pcd)

        if self._first_render:
            self.vis.reset_view_point(True)
            self._first_render = False

        self._clear_boxes()
        boxes = self.pred_boxes[self.idx]
        scores = self.pred_scores[self.idx]
        labels = self.pred_labels[self.idx]
        keep = scores >= self.score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        frame_counts = Counter()
        if boxes.shape[0] > 0:
            corners = boxes_to_corners_3d(boxes)
            for i in range(boxes.shape[0]):
                cls_name = self.class_names[labels[i] - 1]
                frame_counts[cls_name] += 1
                color = DEFAULT_COLOR_MAP.get(cls_name, [1, 0, 1])
                box_geom = make_box_lineset(corners[i], color)
                self.vis.add_geometry(box_geom, reset_bounding_box=False)
                self.box_geoms.append(box_geom)

        breakdown = ', '.join(f'{k}={v}' for k, v in sorted(frame_counts.items())) or 'none'
        total_breakdown = ', '.join(f'{k}={v}' for k, v in sorted(self.total_counts.items()))
        print(f'[frame {self.idx+1}/{self.n}] {fname}  |  this frame: {breakdown}  '
              f'|  running total across all frames: {total_breakdown}')

    def _next(self, vis):
        if self.idx < self.n - 1:
            self.idx += 1
            self._render_frame()
        return False

    def _prev(self, vis):
        if self.idx > 0:
            self.idx -= 1
            self._render_frame()
        return False

    def _quit(self, vis):
        vis.close()
        return False

    def run(self):
        self.vis.run()
        self.vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step through per-frame detections in Open3D')
    parser.add_argument('--npz', type=str, required=True, help='output of infer_frames.py')
    parser.add_argument('--frames_dir', type=str, required=True, help='dir of frame_*.npy from csv_to_frames.py')
    parser.add_argument('--score_thresh', type=float, default=0.0)
    args = parser.parse_args()

    print('Controls: N / Right-arrow = next frame, P / Left-arrow = previous frame, Q = quit')
    stepper = FrameStepper(args.npz, args.frames_dir, args.score_thresh)
    print(f'\nOverall totals across {stepper.n} frames: '
          f'{dict(stepper.total_counts)}\n')
    stepper.run()
