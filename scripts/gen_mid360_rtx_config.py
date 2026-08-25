#!/usr/bin/env python3
"""Generate RTX LiDAR configs for the Livox Mid-360 from its real scan pattern.

Follows the recipe from IsaacSim discussion #685
(https://github.com/isaac-sim/IsaacSim/discussions/685): extract real
per-frame (azimuth, elevation, fireTimeNs, channelId) tuples and encode them
as a sequence of ``emitterStates`` advanced one per frame via
``stateResolutionStep=1``. No artificial elevation binning/line-grouping -
each state carries its points' real, continuously-varying elevation values
directly (numLines=1, one line holding every emitter in the state).

``mid360.npy`` holds 800,000 (theta, phi) pairs. At the real sensor's
200,000 points/s and 10 Hz that is 20,000 points per frame, so the file is
exactly 40 frames of firing pattern.

The discussion's reference example splits one physical sensor into several
co-located prims, each ~5,000 emitters/state, standing in for a subset of
the sensor's beams. We don't have per-beam-labeled data, so each of the 40
real frames is instead split spatially into NUM_PRIMS contiguous chunks of
POINTS_PER_FRAME / NUM_PRIMS points; chunk i always goes to prim i, so all
prims advance through the same 40-frame timeline in sync (each contributing
its slice of the *current* frame), rather than each prim owning a disjoint
span of frames.

Writes one JSON per prim, in the schema of the shipped
``Example_Solid_State.json``:

    python scripts/gen_mid360_rtx_config.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PATTERN = REPO / "assets/scan_patterns/mid360.npy"
DEFAULT_OUT = REPO / "assets/lidar_configs"

# Real Mid-360 figures.
POINTS_PER_SECOND = 200_000
SCAN_RATE_HZ = 10.0
POINTS_PER_FRAME = int(POINTS_PER_SECOND / SCAN_RATE_HZ)  # 20,000

# Matches the discussion's reference example (~5,000 emitters/state/prim).
NUM_PRIMS = 4


def build_emitter_state(chunk: np.ndarray) -> dict:
    """Turn one prim's slice of one frame into an emitterState entry.

    Real per-point azimuth/elevation, in the pattern's own order - no
    binning or elevation quantization. ``fireTimeNs`` spreads the slice's
    points evenly across one scan period; it no longer reflects the
    pattern's real per-point firing order, only the frame-level 100 ms
    period, which is all ``tickRate``-based playback actually relies on.
    """
    azimuth = np.degrees(chunk[:, 0]).astype(np.float64)
    elevation = np.degrees(chunk[:, 1]).astype(np.float64)
    n = len(chunk)

    # RTX expects azimuth in [-180, 180]; the .npy stores [0, 360].
    azimuth = np.where(azimuth > 180.0, azimuth - 360.0, azimuth)

    frame_ns = int(1e9 / SCAN_RATE_HZ)
    fire_time = np.linspace(0, frame_ns, n, endpoint=False).astype(np.int64)

    return {
        "azimuthDeg": [round(v, 4) for v in azimuth.tolist()],
        "elevationDeg": [round(v, 4) for v in elevation.tolist()],
        "fireTimeNs": fire_time.tolist(),
        "channelId": list(range(1, n + 1)),
        "rangeId": [0] * n,
        # Per-emitter line index. numLines=1, so every point is on line 0.
        # The shipped Example_Solid_State.json always authors this array; when
        # it is absent rtx_lidar._profile_to_attributes never writes the
        # omni:sensor:Core:emitterState:sNNN:bank USD attr, and the engine has
        # no per-point line mapping to validate rays against numRaysPerLine.
        "bank": [0] * n,
    }


def build_profile(states: list[dict], model_name: str, max_range: float) -> dict:
    """Assemble the RTX LiDAR profile around a list of emitter states."""
    n_emitters = len(states[0]["azimuthDeg"])
    # Single line holding every emitter in the state - avoids the
    # numLines/numRaysPerLine-vs-bank mismatch that silently dropped most
    # points when lines were used (live-verified 2026-08-11), and the
    # Mid-360 has no real discrete channels to bin into anyway.
    num_rays_per_line = [n_emitters]

    return {
        "class": "sensor",
        "type": "lidar",
        "name": model_name,
        "driveWorksId": "GENERIC",
        "profile": {
            # SOLID_STATE is what makes RTX walk emitterStates rather than
            # synthesise a rotating pattern of its own.
            "scanType": "solidState",
            "intensityProcessing": "normalization",
            "rayType": "IDEALIZED",
            "nearRangeM": 0.1,
            "farRangeM": max_range,
            "rangeResolutionM": 0.004,
            "rangeAccuracyM": 0.02,
            "avgPowerW": 0.002,
            "minReflectance": 0.1,
            "minReflectanceRange": float(max_range),
            "wavelengthNm": 905.0,
            "pulseTimeNs": 6,
            "maxReturns": 1,
            "scanRateBaseHz": SCAN_RATE_HZ,
            "patternFiringRateHz": int(SCAN_RATE_HZ),
            "numberOfEmitters": n_emitters,
            "numberOfChannels": n_emitters,
            "numLines": len(num_rays_per_line),
            "numRaysPerLine": num_rays_per_line,
            "rangeCount": 1,
            "ranges": [{"min": 0.1, "max": max_range}],
            "azimuthErrorMean": 0.0,
            "azimuthErrorStd": 0.015,
            "elevationErrorMean": 0.0,
            "elevationErrorStd": 0.015,
            "intensityMappingType": "LINEAR",
            # Emitter azimuth is converted to [-180, 180] above; match the
            # valid range so RTX does not filter out half the rays.
            "validStartAzimuthDeg": -180.0,
            "validEndAzimuthDeg": 180.0,
            # Advance one emitterState per frame, so the sequence plays in
            # order and wraps - reproducing the non-repetitive sweep.
            "stateResolutionStep": 1,
            "emitterStateCount": len(states),
            "emitterStates": states,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", type=str, default=str(DEFAULT_PATTERN))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--num-prims", type=int, default=NUM_PRIMS)
    parser.add_argument("--max-range", type=float, default=40.0)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Frames to use (0 = all). Fewer frames means a shorter cycle and smaller files.",
    )
    args = parser.parse_args()

    pattern_path = Path(args.pattern)
    if not pattern_path.exists():
        raise SystemExit(f"[GEN] pattern not found: {pattern_path}")

    data = np.load(pattern_path)
    total_frames = len(data) // POINTS_PER_FRAME
    frames = args.frames if args.frames > 0 else total_frames
    frames = min(frames, total_frames)

    emitters_per_prim = POINTS_PER_FRAME // args.num_prims
    if emitters_per_prim == 0:
        raise SystemExit(f"[GEN] {POINTS_PER_FRAME} points/frame cannot fill {args.num_prims} prims")

    print(f"[GEN] pattern      : {pattern_path.name}  ({len(data):,} points)")
    print(f"[GEN] frames       : {frames} of {total_frames} available")
    print(f"[GEN] points/frame : {POINTS_PER_FRAME:,}")
    print(f"[GEN] prims        : {args.num_prims} x {emitters_per_prim:,} emitters/state, {frames} states (synced timeline)")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-shuffle each frame so every prim's chunk covers the full elevation
    # range of the Mid-360 (-7°…+52°). Without this, contiguous chunks from
    # the raw .npy land on elevation-sorted sub-bands (e.g. chunk 0 = upper
    # hemisphere only), causing some prims to see nothing but ceiling and
    # return 0 points. A per-frame shuffle with a fixed seed keeps the split
    # deterministic and reproducible across runs.
    rng = np.random.default_rng(42)
    shuffled = []
    for frame in range(frames):
        frame_data = data[frame * POINTS_PER_FRAME : (frame + 1) * POINTS_PER_FRAME].copy()
        rng.shuffle(frame_data)
        shuffled.append(frame_data)

    written = []
    for prim in range(args.num_prims):
        states = []
        for frame in range(frames):
            chunk = shuffled[frame][prim * emitters_per_prim : (prim + 1) * emitters_per_prim]
            states.append(build_emitter_state(chunk))

        name = f"Livox_Mid360_{chr(ord('A') + prim)}"
        profile = build_profile(states, name, args.max_range)

        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(profile))
        size_mb = path.stat().st_size / 1e6
        written.append((path, size_mb, len(states)))

        # The Hydra API rejects a prim carrying more than ~5 MB of emitter data.
        flag = "OK  " if size_mb < 5.0 else "OVER"
        print(f"[GEN] {flag} {path.name}: {len(states)} states, {size_mb:.2f} MB")

    over = [p for p, mb, _ in written if mb >= 5.0]
    if over:
        print(f"\n[GEN] FAIL: {len(over)} file(s) exceed the 5 MB Hydra limit.")
        print("[GEN] Re-run with more --num-prims or fewer --frames.")
        raise SystemExit(1)

    total = args.num_prims * emitters_per_prim * frames
    print(f"\n[GEN] total emitters: {total:,} across {args.num_prims} prims")
    print(f"[GEN] cycle length  : {frames / SCAN_RATE_HZ:.1f} s")
    print("[GEN] PASS")


if __name__ == "__main__":
    main()
