#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert Livox CSV export to OpenPCDet-compatible .npy point cloud.

OpenPCDet demo.py expects (N, 4) float32 [x, y, z, intensity], intensity in [0, 1].

CSV columns (Livox Viewer export):
Row 0: header (Version,LiDAR Index,...,X,Y,Z,Reflectivity,...)
Row 1: LiDAR serial/metadata row (not data)
Row 2+: actual points

Usage:
    python csv_to_npy.py --csv path/to/file.Csv --out path/to/file.npy
"""

import argparse
import numpy as np
import pandas as pd


def load_csv_points(csv_path):
    df = pd.read_csv(csv_path, header=0, skiprows=[1])
    df.columns = [c.strip() for c in df.columns]
    points = df[['X', 'Y', 'Z', 'Reflectivity']].to_numpy(dtype=np.float32)
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Livox CSV to OpenPCDet .npy')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    out_path = args.out or (args.csv.rsplit('.', 1)[0] + '.npy')

    print(f'Reading {args.csv} ...')
    points = load_csv_points(args.csv)
    print(f'  {points.shape[0]} points')

    # normalize reflectivity (raw, unknown upper bound in this export) to [0, 1]
    refl = points[:, 3]
    refl_max = refl.max()
    if refl_max > 0:
        points[:, 3] = refl / refl_max
    print(f'  intensity range before norm: [{refl.min():.1f}, {refl_max:.1f}]')

    print(f'  xyz range: x[{points[:,0].min():.2f},{points[:,0].max():.2f}] '
          f'y[{points[:,1].min():.2f},{points[:,1].max():.2f}] '
          f'z[{points[:,2].min():.2f},{points[:,2].max():.2f}]')

    np.save(out_path, points)
    print(f'Saved {out_path}')
