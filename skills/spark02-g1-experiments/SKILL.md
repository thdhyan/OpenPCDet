---
name: spark02-g1-experiments
description: "Run G1 Isaac Sim experiments on the DGX Spark aim_spark02 (GB10, aarch64): env-6 LLM social-navigation episodes (named static humans, VoxelNeXt LiDAR detection, gpt-6-luna velocity planner on /g1/cmd_vel with the in-sim WBC), 1080p POV/top/chase video recording, the matching intent-sim 2D run, and Drive/Slides upload via composio. Use when asked to run, record, rebuild or debug anything on spark02 / 'the Spark', or to add an experiment to the G1 social-nav eval. Repo: ~/Projects/thesis/G1_sim (laptop), ~/g1_sim (Spark)."
---

# Spark02 G1 experiments

Full runbook with rationale: `G1_sim/docs/SPARK_SOCIAL_EVAL.md`. This file is the
procedure; read that doc before changing the pipeline itself.

## 0. Ground rules (do not skip)

- **Ask before running any new sim experiment** if the user has said to stop; a
  run costs ~15 min wall and shares a GPU with other people's work.
- **Never touch the base stack**: containers `g1-isaac-sim-spark`,
  `g1-isaac-ros5-perception`, `g1-sim-ws-spark`, `g1-agenticros-rosbridge-spark`
  (ROS domain 42) belong to other sessions. Experiments run as compose project
  `g1eval`, **ROS domain 43**, container suffix `-eval`, cache
  `~/.cache/g1-isaac-sim-eval`.
- **Check memory first**: GB10 memory is unified, `nvidia-smi` shows N/A.
  Use `free -g` (need >= 10 GB available) and
  `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.
- **Never rsync whole directories to Spark.** `~/g1_sim` is an rsync copy (not
  git) that other sessions edit too: `rsync -ain --dry-run` first, diff any file
  whose content differs, then sync only your files with `rsync -aR <files>`.
- **Check status manually** (`docker logs`, probes) - log-pattern monitors sit
  silent on failures that never print. Every failure below was silent.
- API keys: `~/intent-sim/.env` on Spark (mode 600, mounted read-only into the
  planner). Never copy it into an image or a log.

## 1. Layout

```text
laptop  ~/Projects/thesis/{G1_sim, g1_perception_ws, intent-sim}
spark   ~/g1_sim            G1_sim subset (rsync)
        ~/g1_perception_ws  VoxelNeXt/{setup.py,pcdet,tools/cfgs}, LiDAR-HMR/{models,configs,ckpts/humanm3,smplx_models},
                            pt/voxelnext_nuscenes.pth, dapt/pointcept_src/libs/pointops   (rsync -aL: ckpts are symlinks)
        ~/intent-sim        llm/ + .env
```

Images (built on Spark from `docker/docker-compose-isaac-spark.yml`):
`g1-human-detection:spark`, `g1-human-hmr:spark` (target in
`docker/Dockerfile.human-perception-spark`), `g1-llm-planner:spark`.
Rebuild after changing `perception_ros/` (it is COPYed, not mounted):

```bash
ssh aim_spark02 'cd ~/g1_sim && tmux new-session -d -s g1build "docker compose -f docker/docker-compose-isaac-spark.yml --profile humans build human-detection > logs/build_detection.log 2>&1; echo BUILD_EXIT=\$? >> logs/build_detection.log"'
# poll: grep -E "BUILD_EXIT|ERROR" ~/g1_sim/logs/build_detection.log
```

Isaac Sim, `launch/`, `planner/`, `g1_sim/` are bind-mounted: rsync + recreate the container.

## 2. Run one episode

```bash
ssh aim_spark02
cd ~/g1_sim
RUN=run_<task>_$(date +%H%M); echo $RUN > /tmp/g1eval_run
G1_ROS_DOMAIN_ID=43 G1_CONTAINER_SUFFIX=-eval G1_SIM_STEPS=0 \
ISAAC_SIM_CACHE_ROOT=/home/thakk100/.cache/g1-isaac-sim-eval \
G1_SCENE_ENV=social_static G1_NO_PROPS_FLAG= G1_FREEZE_FLAG= G1_NO_LOCO_FLAG= \
G1_RECORD_FLAG="--record-dir logs/eval/$RUN/cams --record-every 12" G1_EVAL_RUN=$RUN \
G1_TASK="Go talk to Dhyan." \
docker compose -p g1eval -f docker/docker-compose-isaac-spark.yml \
  --profile humans --profile planner up -d --force-recreate isaac-sim human-detection llm-planner
```

- Always pass `--profile humans --profile planner` and name all three services
  (`llm-planner` depends on `human-detection`; otherwise "No such service").
- Scene: Dhyan (alone), Zach + Nirshal (talking pair), Stephen - names come from
  USD prim names under `/World/Humans` via `/sim/humans`. Tasks used so far:
  `Go talk to Dhyan.`, `Join Zach and Nirshal's conversation.`
- Planner exits after one episode (`interact`/`listen` or 180 s sim timeout);
  the sim keeps running - stop it yourself.

Progress checks (sim runs at ~0.07x real time next to the base sim; an episode
is ~15 min wall, most of it before the first decision):

```bash
docker logs g1-llm-planner-spark 2>&1 | grep -E "episode|LLM ->" | tail     # decisions / "episode end: {...}"
docker logs g1-human-detection-spark 2>&1 | grep "\[Frame" | tail -2       # VoxelNeXt frames (first appears at Frame 1)
ls ~/g1_sim/logs/eval/$RUN/cams/top | wc -l                                 # recorded frames
```

## 3. Finish: stop, encode, pull, upload

```bash
# on spark, ~30 s of sim after "episode end"
docker stop g1-isaac-sim-spark-eval g1-human-detection-spark
scripts/frames_to_mp4.sh logs/eval/$RUN/cams 5      # 1920x1080, 5 fps = sim time
# laptop
rsync -a --include='*.mp4' --include='trace.json' --include='cams/' --exclude='*' \
  aim_spark02:g1_sim/logs/eval/$RUN/ logs/eval/$RUN/
```

Views: `robot_pov` (head D435, pitched down), `<Name>_pov` (each human's eyes),
`top` (8.5 m, under the 9 m ceiling), `follow` (chase). `trace.json` holds every
observation, LLM call, latency and the scored result.

Upload with composio (`composio whoami` first; Drive + Slides are linked):
- `GOOGLEDRIVE_CREATE_FOLDER {"name", "parent_id"}` per run.
- Videos exceed the 5 MB `GOOGLEDRIVE_UPLOAD_FILE` limit - use
  `composio execute GOOGLEDRIVE_RESUMABLE_UPLOAD --file <mp4> -d '{"folder_to_upload_to": "<id>", "metadata": {"name": "<n>"}, "chunkSize": 8388608}'`.
- Slides: `GOOGLESLIDES_PRESENTATIONS_BATCH_UPDATE` with `markdown_text` appends
  slides; then `createVideo` requests (`source: DRIVE`, the file id, EMU size/transform;
  page is 9144000 x 5143500, place below the title at translateY 1450000).
- Eval folder: https://drive.google.com/drive/folders/1154QfJUDfIjqWzW0PLYKI6aNmtGzjLuW
- Deck: https://docs.google.com/presentation/d/1alHapOXl1MqMxqHvLNtsjbrAREPIJ3XlbckGYFVhr34

## 4. Same scenario in 2D (intent-sim)

```bash
cd ~/Projects/thesis/G1_sim
SDL_VIDEODRIVER=dummy ../intent-sim/.venv/bin/python scripts/env6_2d_intent_sim.py
```

Builds the intent-sim scene from `g1_sim.social_environments.STATIC_HUMANS`,
runs gpt-6-luna (static multi-round) on both tasks, renders 1000x700 MP4s to
`logs/eval/2d/` (kept out of intent-sim's scored `runs/`).

## 5. Silent failures already fixed - check these first

| Symptom | Cause | Fix in place |
|---|---|---|
| detector logs no `[Frame` | sim lidar is `/livox/mid360/points/a` (one topic per prim) | launch input_topic |
| subscriber gets zero clouds | host `rmem_max` 212 KB, ~740 KB clouds fragment; best-effort drops all | `reliable_qos:=true` on bridge + detector |
| bridge TF "extrapolation into the future" | Isaac stamps cloud ~1 us after /tf | fallback to latest TF |
| bridge never publishes | `use_sim_time` + TF wait in its only callback thread | bridge on wall time |
| VoxelNeXt 0 detections, "nvrtc compile failed" | spconv needs EXACT arch match | cumm/spconv built for 12.1 (patched cumm v0.7.11) |
| Triton `sm_121a` ptxas error | Triton too old for GB10 | `TORCHDYNAMO_DISABLE=1` |
| pip "Cannot uninstall numpy 1.26.4" | Debian numpy, ROS depends on it | pin numpy==1.26.4 |
| sim dies at WBC init: no onnxruntime | Isaac Lab image lacks it | `pip --user onnxruntime` into `<cache>/home` (see runbook) |
| sim can't write cams dir | planner (root) created run dir first | planner runs as 1001:1001 |
| top view shows roof | ceiling at 9.0 m | top cam at 8.5 m |
| sim ~0.04x real time | 1080p views rendering every step | views render only 2 steps per capture |
