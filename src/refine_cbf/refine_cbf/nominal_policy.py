# Nominal policy node for refineCBF

# TODO FIND OUT HOW TO GENERALIZE PACKAGE SO THIS CAN BE REMOVED
import sys
sys.path.insert(0, '/home/nate/turtwig_ws/src/refine_cbf/refine_cbf')   # for experiment_utils.py

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
from experiment_utils import *
import logging

data_filename = "/home/nate/turtwig_ws/log/nom_policy_experiment_data.txt"

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

        super().__init__('nominal_policy')

        '''Defining Node Attributes'''

        # Prompt user for configuration
        print('Please select an experiment configuration based on the following series of prompts:')

        # using simulation or real turtlebot3 burger state feedback
        input_sim = input('Use simulation? (y/n): ')

        if input_sim == 'y':
            self.use_simulation = True
        elif input_sim == 'n':
            self.use_simulation = False       
        else:
            raise ValueError('Invalid input')

        # Load nominal policy table
        self.nominal_policy_table = np.load('/home/nate/refineCBF/experiment/data_files/2 by 2 Grid/nominal_policy_table_2x2_coarse_grid_with_bounding_box.npy')

        # Required to give an arbitrary dt to the dynamics object
        # TODO: rewrite dynamics object to not require this as a required argument
        self.dyn = DiffDriveDynamics({"dt": 0.05}, test=False)

        # Environment Parameters

        # defining the state_domain (upper and lower bounds of the state space)
        self.state_domain = hj.sets.Box(lo=jnp.array([0., 0., -jnp.pi]), hi=jnp.array([2., 2., jnp.pi]))

        # define grid resolution as a tuple
        self.grid_resolution = (31, 31, 21)

        # defining the state space grid
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(self.state_domain, self.grid_resolution, periodic_dims=2)

        # initializing common parameters
        self.state = np.array([0.25, 0.25, 0])  # initial state
        self.real_state = np.array([0.25, 0.25, 0]) # initial real state
        self.nominal_policy = np.array([0.1, 0])   # initial nominal policy (used to prevent errors when nominal policy table is not used if command velocity publisher is called before the nominal policy is heard)

        # quality of service profile for subscriber and publisher, provides buffer for messages
        # a depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=10)

        # control publisher
        self.nom_pol_publisher_ = self.create_publisher(
            Twist, 
            'nom_policy',
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
        
        # # real state subscription
        # self.real_state_sub = self.create_subscription(
        #     Odometry,
        #     'odom',
        #     self.real_state_sub_callback,
        #     qos)
        
        # VICON
        # real state subscription
        self.real_state_sub = self.create_subscription(
            TransformStamped,
            '/vicon/turtlebot/turtlebot',
            self.real_state_sub_callback,
            qos)
    
        # prevent unused variable warnings    
        self.state_sub
        self.real_state_sub

        # Data saving and visualization object
        self.data_logger = ParameterStorage()

    '''Callback Functions'''
    
    ##################################################
    ################### PUBLISHERS ###################
    ##################################################

    def timer_callback(self):
        
        # Decide if want to use real state or simulated state
        if self.use_simulation is False:
            self.state = self.real_state # overwrite state with real state

        # Compute Nominal Control
        ############################################

        # If using something other than nominal policy table for nominal policy, set the flag to True
        use_external_nom_policy = False

        if use_external_nom_policy is False:

            # Offline, a nominal policy table was computed for every grid point in the state space.

            # Get value of the nominal policy at the current state using a precomputed nominal policy table
            nominal_policy = self.grid.interpolate(self.nominal_policy_table, self.state)
            nominal_policy = np.reshape(nominal_policy, (1, self.dyn.control_dims))

            print("Nominal Policy: ", nominal_policy)
            # DEBUG: Print the shape and type of the nominal policy for verification
            # print("Nominal Policy Shape: ", np.shape(nominal_policy))
            # print("Nominal policy first element type: ", type(float(nominal_policy[0,0])))
        
        else:
                
            # Use the nominal policy from the external node
            nominal_policy = self.nominal_policy
            # reshape to proper dimensions
            nominal_policy = np.reshape(nominal_policy, (1, self.dyn.control_dims))

        # Publish the Nominal Policy
        ############################################

        # Formulate the message to be published
        msg = Twist()
        msg.linear.x = float(nominal_policy[0,0]) # linear velocity
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(nominal_policy[0,1]) # angular velocity

        # publish the control input
        self.nom_pol_publisher_.publish(msg)
        self.get_logger().info('Publishing nominal control input over topic /nom_policy.')

    ##################################################
    ################### SUBSCRIBERS ##################
    ##################################################

    # Simulation State Subscription
    def state_sub_callback(self, msg):

        # Message to terminal
        self.get_logger().info('Received new simulation tate.')

        # convert quaternion to euler angle
        (roll, pitch, yaw) = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        self.state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        
        print("Current Simulation State: ", self.state)

    # Odometry State Subscription
    # def real_state_sub_callback(self, msg):

    #     # Message to terminal
    #     self.get_logger().info('Received new real state.')

    #     # convert quaternion to euler angle
    #     (roll, pitch, yaw) = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

    #     self.real_state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        
    #     print("Current real State: ", self.real_state)

    # Vicon State Subscription
    def real_state_sub_callback(self, msg):

        # Message to terminal
        self.get_logger().info('Received new real state.')

        # convert quaternion to euler angle
        (roll, pitch, yaw) = euler_from_quaternion(msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w)

        self.real_state = np.array([msg.transform.translation.x, msg.transform.translation.y, yaw])
        
        print("Current real State: ", self.real_state)

def main():

    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

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