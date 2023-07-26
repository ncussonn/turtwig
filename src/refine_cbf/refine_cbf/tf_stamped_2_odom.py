#!/usr/bin/env python3

"""
For use in the Vicon arena.
Converts a topic with a TransformStamped message type to an Odometry message type.
"""

from rclpy.node import Node
from refine_cbf.config import *

class TFStampedToOdom(Node):
    def __init__(self):
        super().__init__('tf_stamped_to_odom')  # Node name in ROS2 network

        # Quality of service profile for subscriber and publisher
        qos = QoSProfile(depth=10)

        # Empty state vector to initialize
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.state_sub = self.create_subscription(
            TransformStamped, STATE_FEEDBACK_TOPIC, self.state_sub_callback_vicon, qos
        )

        self.odom_publisher_ = self.create_publisher(Odometry, 'vicon/odom', qos)

        # Callback timer (how long to wait before running the callback function)
        timer_period = 0.033  # Seconds (equivalent to about ~20Hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        """
        Convert state euler angle into quaternion and shift the quaternion by pi/2 to account for
        the difference between the Vicon and odom frames. Then create an odometry message and publish it.
        """
        original_quaternion = (self.state[5], self.state[2], self.state[3], self.state[4])
        yaw_shift = math.pi / 2  # Yaw rotation of pi/2
        shifted_quaternion = shift_quaternion_by_yaw(original_quaternion, yaw_shift)

        # Create an odometry message
        odom = Odometry()

        # HEADER
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        # POSE
        odom.pose.pose.position.x = self.state[0]
        odom.pose.pose.position.y = self.state[1]
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.x = shifted_quaternion[1]
        odom.pose.pose.orientation.y = shifted_quaternion[2]
        odom.pose.pose.orientation.z = shifted_quaternion[3]
        odom.pose.pose.orientation.w = shifted_quaternion[0]

        # TWIST
        odom.twist.twist.linear.x = 0.0
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0

        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.0

        # Publish the odometry message
        print("Publishing Odometry Message over topic", odom.header.frame_id)
        self.odom_publisher_.publish(odom)

    def state_sub_callback_vicon(self, msg):
        """
        State Subscription callback for TransformStamped. Update the state information from the Vicon arena.
        """
        self.get_logger().info('Received new state information from Vicon arena.')

        x = msg.transform.translation.x
        y = msg.transform.translation.y
        qx = msg.transform.rotation.x
        qy = msg.transform.rotation.y
        qz = msg.transform.rotation.z
        qw = msg.transform.rotation.w

        # Update state
        self.state = np.array([x, y, qx, qy, qz, qw])
        print("Current State: ", self.state)

def main():
    rclpy.init()
    tf_stamped_to_odom = TFStampedToOdom()

    try:
        tf_stamped_to_odom.get_logger().info("Starting TransformStamped to Odometry node, shut down with CTRL+C")
        rclpy.spin(tf_stamped_to_odom)

    except KeyboardInterrupt:
        tf_stamped_to_odom.get_logger().info('Keyboard interrupt, shutting down.\n')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    tf_stamped_to_odom.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    if not HARDWARE_EXPERIMENT:
        print('Hardware experiment is disabled, tf stamped to odom node will not run.')
        exit()

    main()
