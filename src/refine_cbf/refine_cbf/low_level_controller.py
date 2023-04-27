# Low Level Controller Node for RefineCBF

import sys
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
import jax.numpy as jnp
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from experiment_utils import *
import logging

# import proper modules based on the operating system for correct functions and methods 
# for managing console or terminal I/O
# 'nt' is the name of the operating system for Windows
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

class LowLevelController(Node):
    
    '''
    Low Level Controller Node

    - subscribed topics: \gazebo\odom, \odom
    - published topics:  \correctional_control

    '''

    '''Constructor'''
    def __init__(self):

        super().__init__('low_level_controller')

        '''Defining PID Controller Parameters'''
        
        # Gain Parameters
        self.Kp = 1
        self.Kd = 0
        self.Ki = 0

        # Error Parameters
        self.error = 0
        self.error_prev = 0
        self.error_sum = 0

        # Control Parameters
        self.control = 0
        
        '''Defining Node Attributes'''

        # quality of service profile for subscriber and publisher, provides buffer for messages
        # a depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=10)

        # correctional control publisher
        self.control_publisher_ = self.create_publisher(
            Twist, 
            'corrective_control',
            qos)
        
        # callback timer (how long to wait before running callback function)
        timer_period = 0.033 # seconds (equivalent to about ~20Hz, same as odometry/IMU update rate)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # simulation state subscription
        self.state_sub = self.create_subscription(
            Odometry,
            'gazebo/odom',
            self.state_sub_callback,
            qos)
        
        # real state subscription
        self.real_state_sub = self.create_subscription(
            Odometry,
            'odom',
            self.real_state_sub_callback,
            qos)
    
        # prevent unused variable warnings    
        self.state_sub
        self.real_state_sub

    '''Callback Functions'''

    ##################################################
    ################### PUBLISHERS ###################
    ##################################################

    # callback function for timer
    def timer_callback(self):

        # Compute error
        self.error = self.state - self.real_state

        # Compute corrective control using PID
        self.control = self.Kp * self.error + self.Kd * (self.error - self.error_prev) + self.Ki * self.error_sum

        # update previous error
        self.error_prev = self.error

        # update error sum
        self.error_sum += self.error

        # publish control
        self.control_publisher_.publish(self.control)
        
    ##################################################
    ################# SUBSCRIPTIONS ##################
    ##################################################

    # simulation state Subscription
    def state_sub_callback(self, msg):

        # Message to terminal
        self.get_logger().info('Received new simulation tate.')

        # convert quaternion to euler angle
        (roll, pitch, yaw) = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        self.state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        
        print("Current Simulation State: ", self.state)

    # real state Subscription
    def real_state_sub_callback(self, msg):

        # Message to terminal
        self.get_logger().info('Received new real state.')

        # convert quaternion to euler angle
        (roll, pitch, yaw) = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        self.real_state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        
        print("Current real State: ", self.real_state)

def main():

    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()    
    low_level_ctrl = LowLevelController()

    try:
        low_level_ctrl.get_logger().info("Starting saftey filter node, shut down with CTRL+C")
        rclpy.spin(low_level_ctrl)

    except KeyboardInterrupt:
        low_level_ctrl.get_logger().info('Keyboard interrupt, shutting down.\n')

    finally:

        # shut down motors
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        low_level_ctrl.control_publisher_.publish(msg)

        # if on a unix system, restore the terminal settings
        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    low_level_ctrl.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()