# Nominal policy node for refineCBF

# TODO FIND OUT HOW TO GENERALIZE PACKAGE SO THIS CAN BE REMOVED
import sys
#sys.path.insert(0, '/home/nate/refineCBF')
sys.path.insert(0, '/home/nate/refineCBF/experiment')                   # for nominal_hjr_control.py
sys.path.insert(0, '/home/nate/turtwig_ws/src/refine_cbf/refine_cbf')   # for experiment_utils.py

import os
import rclpy
import numpy as np
import jax.numpy as jnp
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
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

    - subscribed topics: \gazebo\odom, \odom
    - published topics:  \nom_policy

    '''

    '''Constructor'''
    def __init__(self):

        super().__init__('safety_filter')

        '''Defining Node Attributes'''

        # Load nominal policy table
        self.nominal_policy_table = np.load('/home/nate/refineCBF/experiment/data_files/2 by 2 Grid/nominal_policy_table_2x2_coarse_grid.npy')

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
      
        ''' Defining parameters for the CBF-QP Solver '''

        # Class K function alpha
        # lambda function: alpha, takes z (in this case the cbf) as an argument, and returns gamma times that argument
        gamma = 0.25 # discount factor (this dictates how fast the system will approach the edge of the safe set)
        self.alpha = lambda z: gamma * z

        # Control Space Constraints
        self.vmin = 0.1      # minimum linear velocity of turtlebot3 burger
        self.vmax = 0.21     # maximum linear velocity of turtlebot3 burger
        self.wmax = 2.63     # maximum angular velocity of turtlebot3 burger
    
        self.umin = np.array([self.vmin, -self.wmax]) # 1x2 array that defines the minimum values the linear and angular velocity can take 
        self.umax = np.array([self.vmax, self.wmax])  # same as above line but for maximum

        self.state = np.array([0.25, 0.25, 0])  # initial state
        self.nominal_policy = np.array([0,0])   # initial nominal policy (used to prevent errors when nominal policy table is not used if command velocity publisher is called before the nominal policy is heard)

        # quality of service profile for subscriber and publisher, provides buffer for messages
        # a depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=10)

        # control publisher
        self.cmd_vel_publisher_ = self.create_publisher(
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
        
        # real state subscription
        self.real_state_sub = self.create_subscription(
            Odometry,
            'odom',
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

        # This callback node needs to be AS FAST AS POSSIBLE. Time delays between control inputs can lead to unsafe behavior.

        # time node
        ctrl_start_time = time.time()
       
        # Compute Nominal Control
        ############################################

        # Offline, a nominal policy table was computed for every grid point in the state space.
        
        # DEBUG: If using nearest grid point, uncomment the following lines - to be removed at later date
        #nearest_grid_point = self.grid.nearest_index(self.state)
        #print(np.shape(nearest_grid_point))
        #print(nearest_grid_point)
        #nominal_policy = self.nominal_policy_table[nearest_grid_point[0], nearest_grid_point[1], nearest_grid_point[2], :]
        #nominal_policy = np.reshape(nominal_policy, (1, self.dyn.control_dims))

        # If using something other than nominal policy table for nominal policy, set the flag to True
        use_external_nom_policy = False

        if use_external_nom_policy is False:

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
        
        # CBF-QP
        ############################################

        # DEBUG: Flag used to determine if nominal control input should be filtered or not. For testing purposes, not to be used in final implementation.
        use_nom_policy = True

        if use_nom_policy is False:

            # Use the Safety Filter

            # Time How Long It Takes To Solve QP
            start_time = time.time()

            # Solve the QP for the optimal control input
            control_input = self.diffdrive_asif(self.state, 0.0, nominal_policy)

            print("CBF-QP solved in %s seconds" % (time.time() - start_time))

            print("Filtered Control Input:", control_input)

            # If the QP solver would return None, record the failure
            if control_input[0].any() == None:
                print("QP solver failed: Returned None")

                # Use logger to log the failure to a file for later analysis
                logging.basicConfig(filename='qp_failure.log', level=logging.DEBUG)
                logging.debug('QP solver failed: Returned None')
                logging.debug('State: "%s"' % self.state)
                logging.debug('Safety Value: "%s"' % self.safety_value)
                logging.debug('Nominal Policy: "%s"' % nominal_policy)

        else:
            # assign unflitered nominal policy as the final control input
            control_input = nominal_policy

        # Publish the Optimal Control Input
        ############################################

        # Formulate the message to be published
        msg = Twist()
        msg.linear.x = float(control_input[0,0]) # linear velocity
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(control_input[0,1]) # angular velocity

        # publish the optimal control input
        self.cmd_vel_publisher_.publish(msg)
        self.get_logger().info('Publishing optimal control input')

        print("Time to run safety filter: %s seconds" % (time.time() - ctrl_start_time))

        # Save visualization data
        ############################################
        self.data_logger.append(x=self.state[0], y=self.state[1], theta=self.state[2], safety_value=self.safety_value, v=control_input[0,0], omega=control_input[0,1])
          
    ##################################################
    ################### SUBSCRIBERS ##################
    ##################################################

    # Boolean flag availability subscription
    def cbf_sub_callback(self, msg):

        self.get_logger().info('New CBF Available: "%s"' % msg.data)

        if msg.data is True:
            
            # Halt the robot to prevent unsafe behavior while the CBF is updated
            self.get_logger().info('New CBF received, stopping robot and loading new CBF')

            # TODO: Figure out how to stop the robot in subscription callback function

            # load new tabular cbf
            self.cbf = jnp.load('./log/cbf.npy')

            # Update the tabular cbf
            self.tabular_cbf.vf_table = np.array(self.cbf)

            # Assign the tabular cbf to the diffdrive asif
            self.diffdrive_asif.cbf = self.tabular_cbf 

    # Simulation State Subscription
    def state_sub_callback(self, msg):

        # Message to terminal
        self.get_logger().info('Received new simulation tate.')

        # convert quaternion to euler angle
        (roll, pitch, yaw) = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        self.state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        
        print("Current Simulation State: ", self.state)

    # Real State Subscription
    def real_state_sub_callback(self, msg):

        # Message to terminal
        self.get_logger().info('Received new real state.')

        # convert quaternion to euler angle
        (roll, pitch, yaw) = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        self.real_state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        
        print("Current real State: ", self.real_state)

        
    # Nominal Policy subscription
    # NOTE: may not be used in final implementation
    def nom_policy_sub_callback(self, msg):

        # Message to terminal
        self.get_logger().info('Received new command.')

        self.nominal_policy = np.array([msg.linear.x, msg.angular.z])
        #self.nominal_policy = np.reshape(self.nominal_policy, (1, self.dyn.control_dims))
        
        
        print("Current Control: ", self.nominal_policy)

def main():

    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()    
    safety_filter = SafetyFilter()

    try:
        safety_filter.get_logger().info("Starting saftey filter node, shut down with CTRL+C")
        rclpy.spin(safety_filter)

    except KeyboardInterrupt:
        safety_filter.get_logger().info('Keyboard interrupt, shutting down.\n')

        # shut down motors
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        safety_filter.cmd_vel_publisher_.publish(msg)

        # save visualization parameters
        safety_filter.data_logger.save_data(data_filename)

        # plot all the data
        safety_filter.data_logger.plot_all()

    finally:

        # shut down motors
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        safety_filter.cmd_vel_publisher_.publish(msg)

        # if on a unix system, restore the terminal settings
        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_filter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()