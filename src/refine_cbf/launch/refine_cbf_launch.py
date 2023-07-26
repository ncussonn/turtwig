from launch import LaunchDescription
from launch_ros.actions import Node
from refine_cbf.config import * # can't import Node here from rclpy.node because it's not a launch action - conflicts with Node from launch_ros.actions

base_node_list = [
            Node(
                package='refine_cbf',
                executable='dynamic_programming',
                name = 'dynamic_programming_node',
            ),
            Node(
                package='refine_cbf',
                executable='refine_cbf_visualization',
                name = 'refine_cbf_visualization_node',
            ),
            Node(
                package='refine_cbf',
                executable='tf_stamped_2_odom',
                name = 'tf_stamped_2_odom_node',
            )]

unfiltered_node_list = base_node_list + [Node(package='refine_cbf', executable='nominal_policy', name = 'nominal_policy_node')]

refine_cbf_node_list = base_node_list + [Node(package='refine_cbf', executable='nominal_policy'), Node(package='refine_cbf', executable='safety_filter', name='safety_filter_node')]

if USE_UNFILTERED_POLICY:
    # refine cbf without safety filter - i.e. only use nominal policy as controls
    def generate_launch_description():
        return LaunchDescription(unfiltered_node_list)

else:
    # default to using refine cbf
    def generate_launch_description():
        return LaunchDescription(refine_cbf_node_list)
