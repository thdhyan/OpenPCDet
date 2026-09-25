# perception_ros — LiDAR human perception for the Spark deployment

Copied from the sister repo `g1_perception_ws` (commit `14b6728`) — only what
the sim stack needs; that repo stays the source of truth for the real robot.

| Package | Node | Container (`docker/Dockerfile.human-perception-spark`) |
|---|---|---|
| `g1_perception` | `lidar_bridge` — `/livox/mid360/points` (inverted `mid360_link`) → `/g1/lidar/points_pelvis` via TF | `human-detection` |
| `livox_detection` | `livox_detection_node` — VoxelNeXt → `/g1/detections/livox` (`Detection3DArray`, `pelvis`) | `human-detection` |
| `g1_perception` | `smpl_hmr_node` — LiDAR-HMR SMPL mesh/β-tracker → `/g1/smpl/*` | `human-hmr` |

`mesh_utils.py` sits at the workspace root because `smpl_hmr_node` imports it
from there. VoxelNeXt/pcdet, LiDAR-HMR sources and all weights are **not**
copied: the image build pulls the sources from `../g1_perception_ws` (compose
`additional_contexts`), and weights are mounted at runtime.

Local changes vs upstream: `lidar_bridge` falls back to the latest TF when the
exact-stamp lookup extrapolates (Isaac Sim stamps clouds ~1 µs after their
/tf); both nodes gained `reliable_qos` (default `false`; the sim launch sets
`true` because the host's 212 KB UDP buffers drop fragments of the ~740 KB sim
clouds, so best-effort readers lose every sample); `livox_detection_node` gained `republish_cloud`
(default `true`, upstream behaviour). The sim sets it `false`, otherwise the node
re-publishes its pelvis-frame input onto the simulator's own
`/livox/mid360/points`.
