#!/usr/bin/env python3
"""Generate the SOLID-STATE (real non-repetitive) Mid-360 RTX config.

Uses the real scan pattern (assets/scan_patterns/mid360.npy) directly:
N consecutive frames, each 20,000 points = one real Mid-360 frame, as
N emitterStates on ONE prim (stateResolutionStep=1 advances per scan).

Scale requirements discovered live 2026-08-25 (see git history):
- states must carry >=~10k emitters: 5k-emitter states emit nothing
- 20k/state fires and equals exactly one real device frame (200k pts/s)
- 5 MiB Hydra per-prim serialization budget: 10 states x 20k x ~20 B
  ~= 4 MB -> keep N <= 12 with the field set written here

Output: assets/lidar_configs_solid/Livox_Mid360_Solid.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PATTERN = REPO / "assets/scan_patterns/mid360.npy"
DEFAULT_OUT = REPO / "assets/lidar_configs_solid"
POINTS_PER_FRAME = 20000
SCAN_RATE_HZ = 10.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=10,
                    help="emitterStates (= frames in the repeat cycle); "
                    "keep <= 12 to stay under the 5 MiB Hydra budget")
    ap.add_argument("--pattern", type=str, default=str(DEFAULT_PATTERN))
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    data = np.load(args.pattern)
    frames = data[: args.frames * POINTS_PER_FRAME].reshape(args.frames, POINTS_PER_FRAME, 2)

    ft = [int(v) for v in np.linspace(0, int(0.999e9 / SCAN_RATE_HZ),
                                      POINTS_PER_FRAME, endpoint=False)]
    states = []
    for f in frames:
        az = np.degrees(f[:, 0])
        az = np.where(az > 180.0, az - 360.0, az)
        el = np.degrees(f[:, 1])
        states.append({
            "azimuthDeg": [round(float(v), 4) for v in az],
            "elevationDeg": [round(float(v), 4) for v in el],
            "fireTimeNs": ft,
            "channelId": list(range(1, POINTS_PER_FRAME + 1)),
            "rangeId": [0] * POINTS_PER_FRAME,
            "bank": [0] * POINTS_PER_FRAME,
        })

    profile = {
        "class": "sensor", "type": "lidar", "name": "Livox_Mid360_Solid",
        "driveWorksId": "GENERIC",
        "profile": {
            "scanType": "solidState",
            "intensityProcessing": "normalization",
            "rayType": "IDEALIZED",
            "nearRangeM": 0.1, "farRangeM": 40.0,
            "rangeResolutionM": 0.004, "rangeAccuracyM": 0.02,
            "minReflectance": 0.1,
            "wavelengthNm": 905.0, "pulseTimeNs": 6,
            "maxReturns": 1,
            "scanRateBaseHz": SCAN_RATE_HZ,
            "patternFiringRateHz": int(SCAN_RATE_HZ),
            "numberOfEmitters": POINTS_PER_FRAME,
            "numberOfChannels": POINTS_PER_FRAME,
            "numLines": 1,
            "numRaysPerLine": [POINTS_PER_FRAME],
            "rangeCount": 1,
            "ranges": [{"min": 0.1, "max": 40.0}],
            "azimuthErrorMean": 0.0, "azimuthErrorStd": 0.015,
            "elevationErrorMean": 0.0, "elevationErrorStd": 0.015,
            "intensityMappingType": "LINEAR",
            "validStartAzimuthDeg": -180.0, "validEndAzimuthDeg": 180.0,
            "stateResolutionStep": 1,
            "emitterStateCount": len(states),
            "emitterStates": states,
        },
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "Livox_Mid360_Solid.json"
    path.write_text(json.dumps(profile))
    mb = path.stat().st_size / 1e6
    print(f"[SOLID] {path}: {len(states)} states x {POINTS_PER_FRAME} real pts, "
          f"{mb:.2f} MB JSON ({'OK' if mb < 5 else 'OVER'} vs 5 MiB serialized budget)")


if __name__ == "__main__":
    main()
