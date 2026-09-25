# Env 6 — LLM social-navigation eval on Spark

Walking G1 (in-sim decoupled WBC on `/g1/cmd_vel`) in the baked warehouse with
four **named, static** humans; VoxelNeXt finds them in the Mid-360 cloud;
`gpt-6-luna` (intent-sim's `LLMClient`) turns named detections into velocity
commands. Everything runs in containers on `aim_spark02`, one Cyclone DDS domain.

```text
isaac-sim (Jazzy, --env social_static)
  /livox/mid360/points, /tf, /clock, /sim/humans ─┐
                                                  ▼
human-detection  /livox/mid360/points/a ─► lidar_bridge ─► /g1/lidar/points_pelvis ─► VoxelNeXt ─► /g1/detections/livox
human-hmr (opt.) LiDAR-HMR ─► /g1/smpl/*                                   │
llm-planner      names ← /sim/humans, robot pose ← /tf World→pelvis ◄───────┘
                 gpt-6-luna every 3 s ─► set_velocity ─► /g1/cmd_vel (20 Hz) ─► WBC
```

## Pieces

| Piece | Where |
|---|---|
| Scene preset 6 (`social_static`) | `g1_sim/social_environments.py` — Dhyan, Zach+Nirshal (talking pair), Stephen at `/World/Humans/<Name>`; T-pose → one-frame "arms down" SkelAnimation; guide-purpose capsule colliders |
| Names from USD | `/sim/humans` (`std_msgs/String` JSON, transient-local, 1 Hz): prim names + world poses, read from the stage |
| Eval cameras | `--record-dir DIR`: 1920×1080 `robot_pov` (head D435), `<Name>_pov` (each human's eyes), `top` (8.5 m, under the 9 m ceiling), `follow` (chase) → `DIR/<cam>/*.jpg`; views render only around capture steps; `scripts/frames_to_mp4.sh DIR 5` (5 fps = sim time at `--record-every 12`) |
| Character assets | `scripts/fetch_people_assets.py` → `assets/people/` (Spark never touches the remote asset root) |
| Detection | `perception_ros/` (copied from `g1_perception_ws@14b6728`), `launch/human_detection_sim.launch.py`, `docker/Dockerfile.human-perception-spark` target `detection` |
| LiDAR-HMR | same Dockerfile, target `hmr`, profile `hmr` (not needed by the planner) |
| Planner | `planner/llm_velocity_planner.py`, `docker/Dockerfile.llm-planner-spark`; mounts `~/intent-sim` read-only for `llm/` + `.env` |

Name association is **oracle**: a detection takes the name of the nearest
roster person within 1 m. Detection itself is the LiDAR's; roster positions
are never shown to the model (it gets robot-frame x/y, range, bearing of the
named detections, plus "last seen" for people who dropped out).

## Spark layout

```text
~/g1_sim            this repo (rsync; not a git checkout)
~/g1_perception_ws  VoxelNeXt/{setup.py,pcdet,tools/cfgs}, LiDAR-HMR/{models,configs,ckpts/humanm3,smplx_models},
                    pt/voxelnext_nuscenes.pth, dapt/pointcept_src/libs/pointops
~/intent-sim        llm/ + .env (chmod 600; OPENAI_API_KEY)
```

## Build (once; spconv/cumm, pcdet ops, torch-scatter/cluster, pointops compile from source)

```bash
cd ~/g1_sim
docker compose -f docker/docker-compose-isaac-spark.yml --profile humans --profile planner build
docker compose -f docker/docker-compose-isaac-spark.yml --profile hmr build   # optional
```

GB10 gotchas baked into `Dockerfile.human-perception-spark`: spconv only uses a
prebuilt kernel on an **exact** arch match, so cumm (patched arch table) and
spconv are built for `12.1`; the NVRTC fallback fails on CUDA 12.9. Triton's
ptxas has no `sm_121a`, so `TORCHDYNAMO_DISABLE=1`. Debian's numpy 1.26.4
cannot be uninstalled by pip (ROS depends on it), so everything is pinned to it.

The Isaac Lab image has no `onnxruntime`, which the in-sim WBC needs (the base
deployment always ran `--no-locomotion`). Install it once into the sim cache's
home, which the container mounts as `$HOME`:

```bash
CACHE=/home/thakk100/.cache/g1-isaac-sim-eval   # cp -a of g1-isaac-sim-base
docker run --rm --user 1001:1001 -e HOME=/home/thakk100 -v $CACHE/home:/home/thakk100 \
  --entrypoint /bin/bash nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 \
  -lc "/isaac-sim/python.sh -m pip install --user onnxruntime"
```

## Run one episode (next to the base sim: own project, ROS domain, cache)

```bash
RUN=run_$(date +%H%M%S)
G1_ROS_DOMAIN_ID=43 G1_CONTAINER_SUFFIX=-eval G1_SIM_STEPS=0 \
ISAAC_SIM_CACHE_ROOT=/home/thakk100/.cache/g1-isaac-sim-eval \
G1_SCENE_ENV=social_static G1_NO_PROPS_FLAG= G1_FREEZE_FLAG= G1_NO_LOCO_FLAG= \
G1_RECORD_FLAG="--record-dir logs/eval/$RUN/cams" G1_EVAL_RUN=$RUN \
G1_TASK="Go talk to Dhyan." \
docker compose -p g1eval -f docker/docker-compose-isaac-spark.yml \
  up -d isaac-sim human-detection llm-planner
docker logs -f g1-llm-planner-spark            # decisions + "episode end: {...}"
scripts/frames_to_mp4.sh logs/eval/$RUN/cams   # follow.mp4, Dhyan.mp4, ...
```

`logs/eval/$RUN/trace.json`: system prompt, every observation/LLM call/latency,
and the result (`success`, target distances at the terminal call, min
clearance to every person, path length). For the next episode recreate
`isaac-sim` and `llm-planner` (the planner exits after one episode).

Tasks that exercise the scene: `Go talk to Dhyan.` · `Join Zach and Nirshal's conversation.` (listen) ·
`Wave to Stephen.` (interact low_wave).

## First results (2026-09-24, "Go talk to Dhyan.", gpt-6-luna)

| Run | Result | Sim time | Decisions | Path | Final dist. | Min clearance to others |
|---|---|---|---|---|---|---|
| `run_dhyan_2308` | success | 12.2 s | 5 | 3.6 m | 0.94 m | 2.18 m |
| `run_dhyan_2334` | success | 21.1 s | 8 | 4.6 m | 1.09 m | 1.85 m |

The sim ran at ~0.07× real time sharing the GPU with a second Isaac Sim, so
LLM latency (~3.5 s wall) is only ~0.25 s of sim time. `run_dhyan_2308`'s top
view is the warehouse roof (camera was above the 9 m ceiling; fixed for 2334).
Videos: `logs/eval/<run>/cams/*.mp4` locally and on Spark, plus Google Drive.
