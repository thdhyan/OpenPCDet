#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split a Livox Mid-360 CSV export into per-frame .npy point clouds, binned by
Timestamp (nanoseconds) at a fixed interval (default 100ms, matching Mid-360's
~10Hz frame-equivalent rate; FrameCounter in these CSVs is not usable — always 0).

Output: <out_dir>/frame_0000.npy, frame_0001.npy, ... each (N,4) float32 [x,y,z,intensity in 0-1]
Also writes <out_dir>/frame_index.csv with frame_id, start_ns, end_ns, num_points, wall_time_s.

Usage:
    python csv_to_frames.py --csv path/to/file.Csv --out_dir path/to/frames --interval_ms 100
"""

import argparse
import os
import numpy as np
import pandas as pd


def load_csv(csv_path):
    df = pd.read_csv(csv_path, header=0, skiprows=[1])
    df.columns = [c.strip() for c in df.columns]
    return df


def split_into_frames(df, interval_ms):
    interval_ns = int(interval_ms * 1_000_000)
    t0 = df['Timestamp'].min()
    frame_id = ((df['Timestamp'] - t0) // interval_ns).astype(np.int64)
    df = df.assign(_frame_id=frame_id)
    return df, t0, interval_ns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split Livox CSV into per-frame .npy point clouds')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--interval_ms', type=float, default=100.0,
                         help='frame bin size in ms (Mid-360 packet rate: 100ms default)')
    parser.add_argument('--min_points', type=int, default=100,
                         help='skip frames with fewer points than this (sparse/edge bins)')
    parser.add_argument('--invert_z', action='store_true',
                         help='negate Z (LiDAR is mounted upside-down on the robot)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Reading {args.csv} ...')
    df = load_csv(args.csv)
    print(f'  {len(df)} points, Timestamp range spans '
          f'{(df["Timestamp"].max() - df["Timestamp"].min())/1e9:.2f}s')

    df, t0, interval_ns = split_into_frames(df, args.interval_ms)
    n_frames = df['_frame_id'].max() + 1
    print(f'  binning into {n_frames} frames @ {args.interval_ms}ms each')

    refl_max = df['Reflectivity'].max()
    refl_max = refl_max if refl_max > 0 else 1.0

    index_rows = []
    n_saved = 0
    for fid, group in df.groupby('_frame_id'):
        if len(group) < args.min_points:
            continue
        pts = group[['X', 'Y', 'Z', 'Reflectivity']].to_numpy(dtype=np.float32)
        pts[:, 3] = pts[:, 3] / refl_max  # normalize intensity to [0,1]
        if args.invert_z:
            pts[:, 2] = -pts[:, 2]  # LiDAR mounted upside-down on robot

        out_path = os.path.join(args.out_dir, f'frame_{fid:05d}.npy')
        np.save(out_path, pts)

        start_ns = t0 + fid * interval_ns
        index_rows.append({
            'frame_id': fid,
            'start_ns': start_ns,
            'end_ns': start_ns + interval_ns,
            'num_points': len(group),
            'wall_time_s': (start_ns - t0) / 1e9,
            'npy_path': out_path,
        })
        n_saved += 1

    index_df = pd.DataFrame(index_rows)
    index_csv = os.path.join(args.out_dir, 'frame_index.csv')
    index_df.to_csv(index_csv, index=False)

    print(f'Saved {n_saved} frames (skipped {n_frames - n_saved} with < {args.min_points} points)')
    print(f'Frame index: {index_csv}')
