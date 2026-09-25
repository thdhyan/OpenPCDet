#!/usr/bin/env bash
# Encode every camera folder written by g1_warehouse_sim.py --record-dir into <cam>.mp4.
#   scripts/frames_to_mp4.sh logs/eval/run_001 [fps]
set -euo pipefail
dir=${1:?record dir}; fps=${2:-5}
for cam in "$dir"/*/; do
  name=$(basename "$cam")
  ls "$cam"*.jpg >/dev/null 2>&1 || continue
  ffmpeg -loglevel error -y -framerate "$fps" -pattern_type glob -i "$cam*.jpg" \
    -c:v libx264 -pix_fmt yuv420p "$dir/$name.mp4"
  echo "$dir/$name.mp4"
done
