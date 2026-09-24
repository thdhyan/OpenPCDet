# Scene presets (`--env`), 2026-09-24

Every preset uses the same G1 (29-DoF + Dex3) with the same sensor suite (Mid-360, D435, 4 IMUs,
fingertip contacts) and the same in-sim WBC. Presets are defined in
`g1_sim/environments.py`; navigation presets are in `g1_sim/nav_environments.py`.

```bash
python -u scripts/g1_warehouse_sim.py --headless --wbc-mode internal --env <name> \
  [--log-props] [--capture-dir DIR] [--caption DIR] [--capture-step N]
```

| `--env` | Scene | IRA | Captures |
|---|---|---|---|
| `tabletop_wheel` (default) | 1. packing table + steering wheel (IsaacLab locomanip pose) | off | 4 views |
| `tabletop_cluster` | 2. table + wheel, red/green/blue cubes, mug, soup can, mustard bottle, banana, foam brick | off | 4 views |
| `nav_people` | 3. warehouse, walking IRA people | on | 6 views |
| `nav_people_boxes` | 4. env 3 + one big and one small box in front of the robot | on | 6 views |
| `nav_people_forearm_box` | 5. env 3 + a box attached to the forearms | on | 6 views |

- `--log-props` prints the PhysX position of every `/World/Props` rigid body every 300 steps.
- `--caption DIR` runs Isaac Sim's VLM Scene Caption on the D435 view at `--capture-step`; see
  `g1_sim/vlm_caption.py`.

## Env 1 — steering wheel fix

- **Root cause:** the wheel URL was `{root}/IsaacLab/Mimic/...`, which returns **HTTP 404**.
  IsaacLab's `ISAACLAB_NUCLEUS_DIR` is `{root}/Isaac/IsaacLab`. An unresolved USD
  reference is only a warning, so the prim was an empty Xform.
- **Second bug, exposed once the reference resolved:** the code applied `RigidBodyAPI` to the reference root.
  The wheel (`Geometry/sm_steeringwheel_a01_01`) and the table tray (`container_h20`)
  already carry their own bodies, so bodies ended up nested and the wheel disappeared. `make_rigid()` now does
  what IsaacLab's `rigid_props` does: it modifies existing bodies, and only adds a body and
  convex-hull colliders to visual-only assets.
- `spawn_usd_prop()` puts the pose on a holder Xform and the reference on `holder/asset`. The asset
  root already authors `double3` xform ops, which clashed with float ops.
- `spawn_usd_prop()` now raises an error if a reference resolves to nothing.
- **Verified:** the wheel stays at (−0.350, 0.450, 0.698) on the table surface (z 0.694) for more than 1000 steps
  and through the WBC walk test, and it is visible in the D435 view (`tabletop_wheel/isaac_d435_rgb.png`).
  `verify_sensor_tf` passes 11/11 and `check_sensor_suite` passes.

## Env 2 — cluster

- YCB `Axis_Aligned` meshes have "up" along −Y inside a Z-up file, so they are rotated −90° about X. With
  +90° the mug was upside down and the mustard bottle stood on its cap and tipped over.
- Props are placed by bbox so they rest 3 mm above the table top (z 0.694). Every prop carries a
  semantic class label, used for segmentation and captioning.
- **Verified:** all 10 objects stay stable for more than 800 steps and are all in the D435 view.
  `verify_sensor_tf` passes 11/11 and `check_sensor_suite` passes.

## VLM Scene Caption (`isaacsim.replicator.caption.core`)

- **Model:** OpenAI `gpt-6-luna` (about 2 s per request). NVIDIA `kimi-k3` took about 67 s,
  `deepseek-v4.1-flash` took more than 150 s, and `cosmos-reason2-8b` and `nemotron-nano` returned 404 for this account.
- **Patches in `vlm_caption.py`:**
  - IRC always sends the `NVIDIA_API_KEY` environment variable, even to other endpoints. The key is unset in-process when an OpenAI endpoint is used.
  - IRC sends `max_tokens`, `chat_template_kwargs`, `temperature` and `top_p`, which gpt-6 rejects, and its health check uses 1 token, which is too few for a reasoning model. These are adapted in-process.
  - IRC merged every labelled prop into a single `/World/Props` node, so only 1 of 10 objects survived. A labelled prim is now treated as its own object.
- **Output:** `<DIR>/World_G1_torso_link_d435_camera/{Image,Data,Captions}`.
  - Env 2 brief caption: *"The scene contains a robot in front of a packing table, along with a
    container, banana, mustard bottle, mug, soup can, foam brick, steering wheel, and red,
    green, and blue cubes. …"*
  - The scene graph has 12 nodes.
## Env 3 — warehouse, walking IRA people (discriminating test)

- **Result:** Robot stands stable pelvis_z ≈ 0.72 m for 400+ updates with IRA humans walking nearby. No fall.
- **Captures:** 6 perspective views saved at --capture-step 300 (`docs/screenshots/envs_20260924/nav_people/`):
  `isaac_front.png`, `isaac_side.png`, `isaac_overview.png`, `isaac_behind.png`, `isaac_robot.png`, `isaac_top.png`.
- **Verification:** `verify_sensor_tf.py` — all depth/color floor checks pass; only `lidar nothing below floor` FAIL (pre-existing, unrelated to fall).
- **Interpretation:** Since env 3 (IRA, no boxes) does not fall, the floor/box obstacles in env 4 are the likely fall cause for env 4. Next: enlarge robot navmesh hole from 0.8×0.8 m to ~1.6 m in `g1_sim/nav_environments.py`.

## Env 1 — steering wheel fix
