#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Headless OpenPCDet inference: no GUI, saves results to .npz, prints tracebacks.
Mirrors demo.py but skips V.draw_scenes (which was masking a SIGFPE crash).

Usage:
    python infer_headless.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt ../checkpoints/pointpillar_7728.pth --data_path <npy_file> --out <out.npz>
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {'points': points, 'frame_id': index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--ext', type=str, default='.npy')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    out_path = args.out or (args.data_path.rsplit('.', 1)[0] + '_pcdet_detections.npz')

    logger = common_utils.create_logger()
    logger.info('----------------- Headless OpenPCDet inference -----------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    logger.info(f'POINT_CLOUD_RANGE: {cfg.DATA_CONFIG.POINT_CLOUD_RANGE}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            logger.info(f'  points after preprocessing: {data_dict["points"].shape}')
            if 'voxels' in data_dict:
                logger.info(f'  voxels: {data_dict["voxels"].shape}')
            if 'voxel_coords' in data_dict:
                vc = data_dict['voxel_coords']
                logger.info(f'  voxel_coords shape: {vc.shape}, min: {vc.min(axis=0)}, max: {vc.max(axis=0)}')
            load_data_to_gpu(data_dict)

            pred_dicts, _ = model.forward(data_dict)

            boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            points_np = data_dict['points'][:, 1:].cpu().numpy()

            logger.info(f'  {boxes.shape[0]} detections')
            class_names = cfg.CLASS_NAMES
            for i in range(boxes.shape[0]):
                logger.info(f'    {class_names[labels[i]-1]:10s}  score={scores[i]:.2f}  '
                             f'xyz=({boxes[i,0]:.1f},{boxes[i,1]:.1f},{boxes[i,2]:.1f})  '
                             f'dims=({boxes[i,3]:.1f},{boxes[i,4]:.1f},{boxes[i,5]:.1f})')

            np.savez(out_path, points=points_np, pred_boxes=boxes, pred_scores=scores,
                     pred_labels=labels, class_names=np.array(class_names))
            logger.info(f'Saved to {out_path}')

    logger.info('Done.')


if __name__ == '__main__':
    main()
