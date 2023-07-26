#!/usr/bin/env python3

# for use in the Vicon arena
# converts a topic with a transform stamped message type to odom message type

from rclpy.node import Node
from refine_cbf.config import *

class TF_Stamped_2_Odom(Node):
        
    def __init__(self):

        # node name in ROS2 network
        super().__init__('tf_stamped_2_odom')

        # a depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=10)

        # empty state vector to initialize
        self.state = np.array([0.,0.,0.,0.,0.,0.])

        self.state_sub = self.create_subscription(
            TransformStamped,
            STATE_FEEDBACK_TOPIC,
            self.state_sub_callback_vicon,
            qos)
        
        self.odom_publisher_ = self.create_publisher(Odometry, 'vicon/odom', qos)

        # callback timer (how long to wait before running callback function)
        timer_period = 0.033 # seconds (equivalent to about ~20Hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # prevent unused variable warnings    
        self.state_sub

    def timer_callback(self):

        # convert state euler angle into quaternion
        #(qx, qy, qz, qw) = quaternion_from_euler(0.0, 0.0, self.state[2])
        # shift the quaternion by pi/2 to account for the difference between the vicon and odom frames

        original_quaternion = (self.state[5],self.state[2], self.state[3], self.state[4])  # Assuming initial quaternion (0, 0, 0, 0)
        
        yaw_shift = math.pi / 2  # Yaw rotation of pi/2

        shifted_quaternion = shift_quaternion_by_yaw(original_quaternion, yaw_shift)

        # create an odometry message
        odom = Odometry()

        ## HEADER
        # set the header of the odometry message
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        ## POSE
        # set the pose of the odometry message
        odom.pose.pose.position.x = self.state[0]
        odom.pose.pose.position.y = self.state[1]
        odom.pose.pose.position.z = 0.0

        # set the orientation of the odometry message
        odom.pose.pose.orientation.x = shifted_quaternion[1]
        odom.pose.pose.orientation.y = shifted_quaternion[2]
        odom.pose.pose.orientation.z = shifted_quaternion[3]
        odom.pose.pose.orientation.w = shifted_quaternion[0]

        ## TWIST
        # set the twist of the odometry message
        odom.twist.twist.linear.x = 0.0
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0

        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.0

        # publish the odometry message
        print("Publishing Odometry Message over topic", odom.header.frame_id)
        self.odom_publisher_.publish(odom)

    # State Subscription for TransformStamped
    def state_sub_callback_vicon(self, msg):

        # Message to terminal
        self.get_logger().info('Received new state information from Vicon arena.')

        x = msg.transform.translation.x
        y = msg.transform.translation.y
        qx = msg.transform.rotation.x
        qy = msg.transform.rotation.y
        qz = msg.transform.rotation.z
        qw = msg.transform.rotation.w

        # update state
        self.state = np.array([x,y,qx,qy,qz,qw])
        
        print("Current State: ", self.state)

def main():

    rclpy.init()    
    tf_stamp_2_odom = TF_Stamped_2_Odom()

    try:
        tf_stamp_2_odom.get_logger().info("Starting transform stamped to odometry node, shut down with CTRL+C")
        rclpy.spin(tf_stamp_2_odom)

    except KeyboardInterrupt:
        tf_stamp_2_odom.get_logger().info('Keyboard interrupt, shutting down.\n')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    tf_stamp_2_odom.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    if HARDWARE_EXPERIMENT is False:
        print('Hardware experiment is disabled, tf stamped to odom node will not run.')
        exit()

    main()
