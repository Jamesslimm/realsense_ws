from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense_node',
            output='screen',
            parameters=[{
                'depth_module.profile': '848x480x30',
                'rgb_camera.profile': '848x480x30',
            }]
        )
    ])