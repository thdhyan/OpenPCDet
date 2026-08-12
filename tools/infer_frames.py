#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run OpenPCDet detection on every frame produced by csv_to_frames.py.
Loads the model once, iterates all frame_*.npy files, saves one combined .npz
with per-frame detections (ragged arrays via object dtype) plus a flat CSV summary.

Usage:
    python infer_frames.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt ../checkpoints/pointpillar_7728.pth \
        --frames_dir path/to/frames --out path/to/detections.npz
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class FramesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, frame_files, training=False, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training,
                          root_path=Path(frame_files[0]).parent, logger=logger)
        self.frame_files = frame_files

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, index):
        points = np.load(self.frame_files[index])
        input_dict = {'points': points, 'frame_id': index}
        return self.prepare_data(data_dict=input_dict)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--frames_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--score_thresh', type=float, default=0.0,
                         help='extra filter on top of model config threshold, for the CSV summary')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    frames_dir = args.frames_dir.rstrip('/')
    out_path = args.out or (frames_dir + '_detections.npz')
    summary_csv = out_path.rsplit('.', 1)[0] + '_summary.csv'

    frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.npy')))
    if not frame_files:
        raise FileNotFoundError(f'No frame_*.npy files found in {frames_dir}')

    logger = common_utils.create_logger()
    logger.info(f'Found {len(frame_files)} frames in {frames_dir}')

    dataset = FramesDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                             frame_files=frame_files, training=False, logger=logger)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    class_names = cfg.CLASS_NAMES
    per_frame_boxes, per_frame_scores, per_frame_labels = [], [], []
    summary_rows = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            data_dict = dataset.collate_batch([dataset[idx]])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            keep = scores >= args.score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            per_frame_boxes.append(boxes)
            per_frame_scores.append(scores)
            per_frame_labels.append(labels)

            frame_name = os.path.basename(frame_files[idx])
            for i in range(boxes.shape[0]):
                summary_rows.append({
                    'frame_file': frame_name,
                    'frame_idx': idx,
                    'class': class_names[labels[i] - 1],
                    'score': scores[i],
                    'x': boxes[i, 0], 'y': boxes[i, 1], 'z': boxes[i, 2],
                    'dx': boxes[i, 3], 'dy': boxes[i, 4], 'dz': boxes[i, 5],
                    'heading': boxes[i, 6],
                })

            if (idx + 1) % 20 == 0 or idx == len(dataset) - 1:
                logger.info(f'  [{idx+1}/{len(dataset)}] {frame_name}: {boxes.shape[0]} detections')

    np.savez(out_path,
              frame_files=np.array([os.path.basename(f) for f in frame_files]),
              pred_boxes=np.array(per_frame_boxes, dtype=object),
              pred_scores=np.array(per_frame_scores, dtype=object),
              pred_labels=np.array(per_frame_labels, dtype=object),
              class_names=np.array(class_names))
    logger.info(f'Saved per-frame detections to {out_path}')

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f'Saved flat summary ({len(summary_df)} detections across all frames) to {summary_csv}')


if __name__ == '__main__':
    main()
