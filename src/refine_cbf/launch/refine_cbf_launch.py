import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='refine_cbf',
            executable='safety_filter',
        ),
        Node(
            package='refine_cbf',
            executable='nominal_policy',
        ),
        Node(
            package='refine_cbf',
            executable='dynamic_programming',
        ),
        Node(
            package='refine_cbf',
            executable='refine_cbf_visualization',
        ),
    ])
