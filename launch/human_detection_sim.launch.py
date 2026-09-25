"""VoxelNeXt human detection on the simulated Mid-360 (human-detection container).

/livox/mid360/points/a (mid360_link, mounted inverted; the sim publishes one
topic per lidar prim, g1_sim/rtx_publisher.py)
    -> lidar_bridge (TF -> pelvis)   -> /g1/lidar/points_pelvis
    -> livox_detection_node          -> /g1/detections/livox (Detection3DArray, pelvis)

offset_ground shifts the pelvis-frame cloud so the floor sits where VoxelNeXt's
nuScenes training data has it (~1.8 m below the sensor); the pelvis stands
~0.75 m above the floor, hence the -1.05 default.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = {
        "checkpoint_path": "/checkpoints/voxelnext_nuscenes.pth",
        "voxelnext_dir": "/ws/VoxelNeXt",
        "score_threshold": "0.15",
        "offset_ground": "-1.05",
        "accumulate_frames": "4",
        "max_hz": "5.0",
        "max_distance": "12.0",
    }
    cfg = {k: LaunchConfiguration(k) for k in args}
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
        + [
            Node(
                package="g1_perception",
                executable="lidar_bridge",
                name="g1_lidar_bridge",
                output="screen",
                parameters=[{
                    "input_topic": "/livox/mid360/points/a",
                    "target_frame": "pelvis",
                    "output_topic": "/g1/lidar/points_pelvis",
                    "reliable_qos": True,  # host UDP buffers drop fragments of the ~740 KB clouds
                    # wall time: its TF wait runs inside its only callback thread, so on
                    # sim time (slow /clock) the timeout never expires and nothing is published
                    "use_sim_time": False,
                }],
            ),
            Node(
                package="livox_detection",
                executable="livox_detection_node",
                name="g1_livox_detection",
                output="screen",
                parameters=[{
                    "algorithm": "voxelnext",
                    "checkpoint_path": cfg["checkpoint_path"],
                    "voxelnext_dir": cfg["voxelnext_dir"],
                    "voxelnext_cfg": "/ws/VoxelNeXt/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml",
                    "input_topic": "/g1/lidar/points_pelvis",
                    "target_frame": "pelvis",
                    "republish_cloud": False,
                    "reliable_qos": True,
                    "class_filter": "pedestrian",
                    "score_threshold": cfg["score_threshold"],
                    "offset_ground": cfg["offset_ground"],
                    "accumulate_frames": cfg["accumulate_frames"],
                    "max_hz": cfg["max_hz"],
                    "max_distance": cfg["max_distance"],
                    "use_sim_time": True,
                }],
            ),
        ]
    )
