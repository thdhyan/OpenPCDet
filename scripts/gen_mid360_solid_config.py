#!/usr/bin/env python3
"""Generate the SOLID-STATE (real non-repetitive) Mid-360 RTX config.

Uses the real scan pattern (assets/scan_patterns/mid360.npy): ONE emitter
state holding the first recorded frame (20,000 points = one real Mid-360
frame at 10 Hz). At runtime g1_sim.rtx_lidar.ScanPatternCycler rewrites that
state with the next recorded frame every scan, so the pattern never repeats
within the recording.

Engine constraints (live-measured):
- states need >=~10k emitters: 5k-emitter states emit nothing (2026-08-25)
- multiple emitter states squeeze elevation: 10 states returned -20..10 deg
  for an authored -7..52 deg; a single state reproduces it (2026-09-23)
- azimuth is swept clockwise, handled by rtx_lidar.pattern_frame_deg

Emitters are grouped into scan lines (bank = line index) following NVIDIA's
arbitrary-pattern Blickfeld Cube 1 profile.

Output: assets/lidar_configs_solid/Livox_Mid360_Solid.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from g1_sim.rtx_lidar import POINTS_PER_FRAME, SCAN_PATTERN, SCAN_RATE_HZ, pattern_frame_deg  # noqa: E402

DEFAULT_OUT = REPO / "assets/lidar_configs_solid"
NUM_LINES = 100
RAYS_PER_LINE = POINTS_PER_FRAME // NUM_LINES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", type=str, default=str(SCAN_PATTERN))
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    frame = np.load(args.pattern)[:POINTS_PER_FRAME]
    az, el = pattern_frame_deg(frame)
    state = {
        "azimuthDeg": [round(float(v), 4) for v in az],
        "elevationDeg": [round(float(v), 4) for v in el],
        "fireTimeNs": [int(v) for v in np.linspace(0, int(0.999e9 / SCAN_RATE_HZ), POINTS_PER_FRAME, endpoint=False)],
        "channelId": list(range(1, POINTS_PER_FRAME + 1)),
        "rangeId": [0] * POINTS_PER_FRAME,
        "bank": [i // RAYS_PER_LINE for i in range(POINTS_PER_FRAME)],
    }

    profile = {
        "class": "sensor", "type": "lidar", "name": "Livox_Mid360_Solid",
        "driveWorksId": "GENERIC",
        "profile": {
            "scanType": "solidState",
            "intensityProcessing": "normalization",
            "rayType": "IDEALIZED",
            "nearRangeM": 0.1, "farRangeM": 40.0,
            "rangeResolutionM": 0.004, "rangeAccuracyM": 0.02,
            "minReflectance": 0.02,
            "wavelengthNm": 905.0, "pulseTimeNs": 6,
            "maxReturns": 1,
            "scanRateBaseHz": SCAN_RATE_HZ,
            "patternFiringRateHz": int(SCAN_RATE_HZ),
            "numberOfEmitters": POINTS_PER_FRAME,
            "numberOfChannels": POINTS_PER_FRAME,
            "numLines": NUM_LINES,
            "numRaysPerLine": [RAYS_PER_LINE] * NUM_LINES,
            "rangeCount": 1,
            "ranges": [{"min": 0.1, "max": 40.0}],
            "azimuthErrorMean": 0.0, "azimuthErrorStd": 0.015,
            "elevationErrorMean": 0.0, "elevationErrorStd": 0.015,
            "intensityMappingType": "LINEAR",
            "validStartAzimuthDeg": -180.0, "validEndAzimuthDeg": 180.0,
            "stateResolutionStep": 1,
            "emitterStateCount": 1,
            "emitterStates": [state],
        },
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "Livox_Mid360_Solid.json"
    path.write_text(json.dumps(profile))
    print(f"[SOLID] {path}: 1 state x {POINTS_PER_FRAME} real pts ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
