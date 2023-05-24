import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

import sys
sys.path.insert(0, '/home/nate/turtwig_ws/src/refine_cbf/refine_cbf') # for config file
from config import *

if USE_MANUAL_CONTROLLER:

    def generate_launch_description():
        return LaunchDescription([
            Node(
                package='refine_cbf',
                executable='safety_filter',
            ),
            Node(
                package='refine_cbf',
                executable='dynamic_programming',
            ),
            Node(
                package='refine_cbf',
                executable='refine_cbf_visualization',
            ),
            # Node(
            #     package='refine_cbf',
            #     executable='teleop_keyboard',
            # ),            
        ])
    
else:
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
