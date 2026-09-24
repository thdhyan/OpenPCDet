"""Launch CUVSLAM (visual SLAM) + NVBLOX (3D reconstruction) inside the Isaac Sim container.

Wires the sim's D435 camera topics to Isaac ROS perception nodes and publishes
all output topics through zenoh for remote visualization.

Sim sensor topics consumed:
    /g1/camera/rgb           sensor_msgs/Image        (BGR8)
    /g1/camera/depth         sensor_msgs/Image        (32FC1 depth in metres)
    /g1/camera/camera_info   sensor_msgs/CameraInfo
    /g1/imu                  sensor_msgs/Imu
    /tf                      (World -> pelvis -> robot chain)

CUVSLAM output:
    /tf                      map -> odom -> pelvis
    /visual_slam/tracking/odometry   nav_msgs/Odometry
    /visual_slam/tracking/slam_path  nav_msgs/Path

NVBLOX output:
    /nvblox/gpu_mesh              visualization_msgs/MarkerArray
    /nvblox/esdf_distance_slice   nav_msgs/OccupancyGrid
    /nvblox/costmap               nav_msgs/OccupancyGrid

Run inside the container:
    ros2 launch /workspace/thesis-sim/launch/perception_launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("log_level", default_value="info",
                              choices=["debug", "info", "warn", "error"]),

        LogInfo(msg=["Launching CUVSLAM + NVBLOX perception stack"]),

        # ── 1a. Static TF: World → map ────────────────────────────────────
        # Bridges the sim's World tree into CUVSLAM's map tree so NVBLOX can
        # trace: map → odom → pelvis → ... → d435_color_optical_frame
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="world_to_map",
            arguments=["0", "0", "0", "0", "0", "0", "World", "map"],
            parameters=[{"use_sim_time": True}],
        ),

        # ── 1b. TF fallback: map → pelvis (on /tf, NOT /tf_static) ─────────
        # NVBLOX subscribes to /tf for frame lookups, but static_transform_
        # publisher only publishes on /tf_static. This Python node uses
        # TransformBroadcaster to publish map→pelvis on /tf so NVBLOX can
        # always find the pelvis frame. CUVSLAM's higher-frequency TF overrides
        # this when tracking.
        ExecuteProcess(
            cmd=["python3",
                 "/workspace/thesis-sim/launch/tf_fallback.py",
                 "--ros-args", "-p", "use_sim_time:=true"],
            output="screen",
        ),

        # ── 2. Depth format converter (32FC1 m → uint16 mm) ───────────────
        # CUVSLAM RGBD mode expects uint16 depth in mm; Isaac Sim publishes
        # 32FC1 in metres. This small Python node converts on the fly.
        ExecuteProcess(
            cmd=["python3",
                 "/workspace/thesis-sim/G1_sim/launch/depth_converter.py"],
            output="screen",
        ),

        # ── 3. Composable node container (CUVSLAM + NVBLOX) ───────────────
        ComposableNodeContainer(
            name="perception_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container_isolated",
            composable_node_descriptions=[

                # ── CUVSLAM visual SLAM (RGBD mode) ───────────────────────
                ComposableNode(
                    name="visual_slam_node",
                    package="isaac_ros_visual_slam",
                    plugin="nvidia::isaac_ros::visual_slam::VisualSlamNode",
                    remappings=[
                        ("visual_slam/image_0", "/g1/camera/rgb"),
                        ("visual_slam/camera_info_0", "/g1/camera/camera_info"),
                        ("visual_slam/depth_0", "/g1/camera/depth_uint16"),
                        ("visual_slam/imu", "/g1/imu"),
                    ],
                    parameters=[{
                        "use_sim_time": True,

                        # Camera
                        "num_cameras": 1,
                        "min_num_images": 1,
                        "rectified_images": False,
                        "enable_image_denoising": False,

                        # Tracking: 0=multi-cam, 1=VIO, 2=RGBD
                        "tracking_mode": 2,
                        "depth_scale_factor": 1000.0,   # uint16 mm → metres
                        "depth_camera_id": 0,
                        "depth_enable_stereo_tracking": False,

                        # IMU noise (nominal — D435 has no IMU)
                        "gyro_noise_density": 0.000244,
                        "gyro_random_walk": 0.000019393,
                        "accel_noise_density": 0.001862,
                        "accel_random_walk": 0.003,
                        "calibration_frequency": 200.0,

                        # Frames
                        "camera_optical_frames": ["d435_color_optical_frame"],
                        "imu_frame": "imu_in_torso",
                        "base_frame": "pelvis",

                        # SLAM
                        "enable_localization_n_mapping": True,
                        "map_frame": "map",
                        "odom_frame": "odom",

                        # Visualization
                        "enable_slam_visualization": True,
                        "enable_landmarks_view": True,
                        "enable_observations_view": True,
                        "path_max_size": 200,
                        "verbosity": 3,
                    }],
                ),

                # ── NVBLOX 3D reconstruction ───────────────────────────────
                ComposableNode(
                    name="nvblox_node",
                    package="nvblox_ros",
                    plugin="nvblox::NvbloxNode",
                    remappings=[
                        ("camera_0/depth/image", "/g1/camera/depth"),
                        ("camera_0/depth/camera_info", "/g1/camera/camera_info"),
                        ("camera_0/color/image", "/g1/camera/rgb"),
                        ("camera_0/color/camera_info", "/g1/camera/camera_info"),
                    ],
                    parameters=[{
                        "use_sim_time": True,

                        # Input
                        "num_cameras": 1,
                        "use_tf_transforms": True,
                        "input_qos": "SENSOR_DATA",

                        # Voxel grid
                        "voxel_size": 0.05,
                        "mapping_type": "static_tsdf",

                        # Processing rates
                        "tick_period_ms": 10,
                        "integrate_depth_rate_hz": 40.0,
                        "integrate_color_rate_hz": 5.0,
                        "update_mesh_rate_hz": 5.0,
                        "update_esdf_rate_hz": 10.0,
                        "publish_layer_rate_hz": 5.0,

                        # ESDF
                        "esdf_mode": "2d",
                        "publish_esdf_distance_slice": True,

                        # Sensor
                        "use_color": True,
                        "use_depth": True,
                        "use_lidar": False,

                        # Map clearing
                        "map_clearing_radius_m": 15.0,
                        "map_clearing_frame_id": "pelvis",

                        # Integrator
                        "static_mapper.projective_integrator_max_integration_distance_m": 10.0,
                        "static_mapper.projective_integrator_truncation_distance_vox": 4.0,
                        "static_mapper.projective_integrator_weighting_mode":
                            "inverse_square_tsdf_distance_penalty",
                        "static_mapper.projective_integrator_max_weight": 5.0,

                        # ESDF
                        "static_mapper.esdf_integrator_min_weight": 0.1,
                        "static_mapper.esdf_integrator_max_site_distance_vox": 2.0,
                        "static_mapper.esdf_integrator_max_distance_m": 2.0,
                        "static_mapper.esdf_slice_height": 0.09,
                        "static_mapper.esdf_slice_min_height": 0.09,
                        "static_mapper.esdf_slice_max_height": 0.65,

                        # Workspace (warehouse-sized)
                        "static_mapper.workspace_bounds_type": "height_bounds",
                        "static_mapper.workspace_bounds_min_corner_x_m": -15.0,
                        "static_mapper.workspace_bounds_min_corner_y_m": -15.0,
                        "static_mapper.workspace_bounds_min_height_m": -0.5,
                        "static_mapper.workspace_bounds_max_corner_x_m": 15.0,
                        "static_mapper.workspace_bounds_max_corner_y_m": 15.0,
                        "static_mapper.workspace_bounds_max_height_m": 3.0,

                        # Mesh streamer
                        "static_mapper.layer_streamer_exclusion_height_m": 2.5,
                        "static_mapper.layer_streamer_exclusion_radius_m": 15.0,

                        # Visualization
                        "esdf_slice_bounds_visualization_attachment_frame_id": "pelvis",
                        "esdf_slice_bounds_visualization_side_length": 15.0,
                        "workspace_height_bounds_visualization_attachment_frame_id": "pelvis",
                        "workspace_height_bounds_visualization_side_length": 15.0,
                    }],
                ),
            ],
            output="screen",
            arguments=["--ros-args", "--log-level", log_level],
        ),
    ])
