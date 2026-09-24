# Foundation Model Debug UI

Web UI for inspecting GR00T, Cosmos3-Edge, and Cosmos3-Nano inputs and outputs.

![Foundation Model Debug UI](screenshots/fm-debug-urdf-preview.png)

*Captured locally on 2026-09-24 using the offline demo trajectory. This is the physics-disabled URDF preview; the disconnected indicator is expected until the Isaac Sim and model WebSocket services are running.*

## Quick Start

```bash
# 1. Install dependencies
pip install aiohttp websockets numpy

# 2. Start the UI server
python fm_debug_ui/server.py --port 8080

# 3. Open in browser
# http://localhost:8080
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    BROWSER (ComfyUI-style UI)                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ Model Select │  │ Input/Output     │  │ 3D URDF View      │  │
│  │ Schema Select│  │ Camera Feed      │  │ Joint Angles      │  │
│  │ Connection   │  │ Action Output    │  │ EEF SE(3) Poses   │  │
│  └──────────────┘  └──────────────────┘  └───────────────────┘  │
└───────────────────────────┬──────────────────────────────────────┘
                            │ WebSocket
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    UI SERVER (aiohttp, :8080)                     │
│  /ws/model → proxy to GR00T (:8765) / Cosmos (:8770/:8771)      │
│  /ws/sim   → proxy to Isaac Sim bridge (:8766)                  │
│  /api/ik   → IK solver (SE3 → joint angles)                     │
└──────────────────────────────────────────────────────────────────┘
```

## Available Models

| Model | Type | WS Endpoint | Output |
|-------|------|-------------|--------|
| GR00T N1.7-3B | Action | ws://localhost:8765 | Joint angles, SE3 |
| Cosmos3-Edge | Video | ws://localhost:8770 | Next image/video |
| Cosmos3-Nano | Video | ws://localhost:8771 | Next image/video |

## Output Schemas

- **Upper Body Action** — 14 arm joint angles
- **Full Body Action** — 29 joint angles + base commands
- **EEF SE(3)** — 6D end-effector poses
- **Joint Angles** — raw joint values
- **Next Image** — predicted RGB+depth
- **Next Video** — predicted video frames
- **Language Output** — generated text

## Connecting to Isaac Sim

Start the warehouse scene and the simulator bridge first; the browser connects
through this server's WebSocket proxies.

```bash
# Isaac Sim host / GPU container
python scripts/g1_warehouse_sim.py --wbc-mode internal --no-ira --config-dir assets/lidar_configs_rotary

# Run this in the same ROS 2 / Isaac Sim environment as the simulator
python scripts/sim_ws_bridge.py --port 8766 --rate-hz 10

# Model host (for example, spark02)
python scripts/gr00t_ws_server.py --model ~/foundation_models/GR00T-N1.7-3B --port 8765
```

In the browser, the defaults are:

- Model proxy: `ws://localhost:8080/ws/model?model=gr00t`
- Simulator proxy: `ws://localhost:8080/ws/sim?sim_url=ws://localhost:8766`

The model response now includes the complete arm trajectory:

```json
{
  "type": "arm_cmd",
  "name": ["left_shoulder_pitch_joint", "..."],
  "position": [/* first absolute pose */],
  "trajectory": [/* T × 14 absolute poses */],
  "horizon": 40,
  "preview_only": true,
  "dispatch": "explicit_ui_approval_required"
}
```

The 3D tab loads `assets/g1_29dof.urdf` with Three.js in the browser. It is a
**physics-disabled visual URDF**; the slider previews the predicted trajectory.
The **Dispatch selected pose** button is the only UI path that sends the
selected arm pose through the simulator bridge to `/g1/arm_cmd`.

For a quick offline UI check without a model or simulator:

```text
http://localhost:8080/?demo=1&view=3d-view
```

The direct ROS bridge is preview-only by default. If that older path is used,
`scripts/ws_sensor_bridge.py` requires the explicit `--send-to-sim` flag before
it publishes model actions.

The browser module imports Three.js and `urdf-loader` from jsDelivr. The
simulator/model server does not need internet access, but the browser must be
able to reach the CDN unless those two frontend packages are vendored later.

The server also exposes `GET /api/models`, `GET /api/schema`,
`GET /api/robot/g1-29dof`, `GET /assets/g1_29dof.urdf`, and `POST /api/ik`.

