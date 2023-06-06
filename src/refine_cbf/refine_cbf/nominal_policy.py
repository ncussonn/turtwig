# Nominal policy node for refineCBF

# TODO FIND OUT HOW TO GENERALIZE PACKAGE SO THIS CAN BE REMOVED
import sys
sys.path.insert(0, '/home/nate/turtwig_ws/src/refine_cbf/refine_cbf')   # for experiment_utils.py and config.py

import os
import rclpy
import numpy as np
import jax.numpy as jnp
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
import refine_cbfs
from refine_cbfs.dynamics import HJControlAffineDynamics 
import hj_reachability as hj
import time
from config import *
import logging

# import proper modules based on the operating system for correct functions and methods 
# for managing console or terminal I/O
# 'nt' is the name of the operating system for Windows
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

class NominalPolicy(Node):

    '''
    Noimnal Policy Node

    - subscribed topics: \gazebo\odom, \odom, \vicon_odom
    - published topics:  \nom_policy

    '''

    '''Constructor'''
    def __init__(self):

        # name of node for ROS2
        super().__init__('nominal_policy')

        '''Defining Node Attributes'''

        # Load nominal policy table
        self.nominal_policy_table = np.load(NOMINAL_POLICY_FILENAME)

        # Required to give an arbitrary dt to the dynamics object
        # TODO: rewrite dynamics object to not require this as a required argument
        self.dyn = DiffDriveDynamics({"dt": 0.05}, test=False)

        # Environment Parameters

        # defining the state_domain (upper and lower bounds of the state space)
        self.state_domain = hj.sets.Box(lo=jnp.array([0., 0., -jnp.pi]), hi=jnp.array([2., 2., jnp.pi]))

        # define grid resolution as a tuple
        self.grid_resolution = (61, 61, 61)

        # defining the state space grid
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(self.state_domain, self.grid_resolution, periodic_dims=2)

        # initializing common parameters
        self.state = INITIAL_STATE  # initial state
        self.nominal_policy = np.array([0.1, 0])   # initial nominal policy (used to prevent errors when nominal policy table is not used if command velocity publisher is called before the nominal policy is heard)

        # quality of service profile for subscriber and publisher, provides buffer for messages
        # a depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=NOMINAL_POLICY_QOS_DEPTH)

        # control publisher
        self.nom_pol_publisher_ = configure_nominal_policy_publisher(self, qos, USE_UNFILTERED_POLICY)
        
        # callback timer (how long to wait before running callback function)
        timer_period = NOMINAL_POLICY_TIMER_SECONDS # 0.033 # seconds (equivalent to about ~20Hz, same as odometry/IMU update rate, any higher is pointless additional computing resources)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # STATE FEEDBACK

        # configure the state subscriber based on the state feedback source (simulation, real turtlebot3 burger, or vicon)
        self.state_sub = configure_state_feedback_subscriber(self, qos, STATE_FEEDBACK_TOPIC)
    
        # prevent unused variable warnings    
        self.state_sub

        # Data saving and visualization object
        self.data_logger = ParameterStorage()

    '''Callback Functions'''
    
    ##################################################
    ################### PUBLISHERS ###################
    ##################################################

    def timer_callback(self):
        
        # Compute Nominal Control
        ############################################

        # # If using something other than nominal policy table for nominal policy, set the flag to True
        # use_external_nom_policy = False

        # Offline, a nominal policy table was computed for every grid point in the state space.

        # Get value of the nominal policy at the current state using a precomputed nominal policy table

        self.nominal_policy = compute_nominal_control(self)

        # swap sign of angular velocity to match turtlebot3 convention
        # nominal_policy = nominal_policy.at[1].set(-nominal_policy[0,1])

        # nominal_policy = self.grid.interpolate(self.nominal_policy_table, self.state)
        # nominal_policy = np.reshape(nominal_policy, (1, self.dyn.control_dims))

        print("Nominal Policy: ", self.nominal_policy)
        
        # Publish the Nominal Policy
        ############################################

        # Formulate the message to be published
        msg = Twist()
        msg.linear.x = float(self.nominal_policy[0,0]) # linear velocity
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(self.nominal_policy[0,1]) # angular velocity

        # publish the control input
        self.nom_pol_publisher_.publish(msg)
        publish_message = create_nominal_policy_publishing_message(USE_UNFILTERED_POLICY)
        self.get_logger().info(publish_message)

        # Save visualization data
        ############################################
        self.data_logger.append(x=self.state[0], y=self.state[1], theta=self.state[2], v_nom=self.nominal_policy[0,0], omega_nom=self.nominal_policy[0,1])

    ##################################################
    ################### SUBSCRIBERS ##################
    ##################################################

    # Simulation State Subscription
    def state_sub_callback(self, msg):

        # Message to terminal
        self.get_logger().info('Received new state information.')

        # convert quaternion to euler angle
        (roll, pitch, yaw) = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        self.state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        
        print("Current State: ", self.state)

    # State Subscription for TransformStamped
    def state_sub_callback_vicon(self, msg):

        # Message to terminal
        self.get_logger().info('Received new state information from Vicon arena.')

        # convert quaternion to euler angle
        (roll, pitch, yaw) = euler_from_quaternion(msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w)

        self.state = np.array([msg.transform.translation.x, msg.transform.translation.y, yaw+np.pi/2])
        
        print("Current State: ", self.state)

def main():

    if USE_MANUAL_CONTROLLER is True:
        print('Manual controller is enabled, nominal policy node will not run.')
        exit()

    settings = None
    # if os.name != 'nt':
    #     settings = termios.tcgetattr(sys.stdin)

    rclpy.init()    
    nominal_policy = NominalPolicy()

    try:
        nominal_policy.get_logger().info("Starting saftey filter node, shut down with CTRL+C")
        rclpy.spin(nominal_policy)

    except KeyboardInterrupt:
        nominal_policy.get_logger().info('Keyboard interrupt, shutting down.\n')

        # shut down motors
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        nominal_policy.nom_pol_publisher_.publish(msg)
        nominal_policy.data_logger.save_data(DATA_FILENAME_NOMINAL_POLICY)
        print("Data saved to: ", DATA_FILENAME_NOMINAL_POLICY)

    finally:

        # shut down motors
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        nominal_policy.nom_pol_publisher_.publish(msg)

        # if on a unix system, restore the terminal settings
        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    nominal_policy.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()