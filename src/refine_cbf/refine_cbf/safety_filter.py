#!/usr/bin/env python3

# Subscribes to Boolean Publisher and State Publisher
# Publishes Twist Message

from rclpy.node import Node
from refine_cbf.config import *

class SafetyFilter(Node):

    '''
    Safety Filter Node

    - subscribed topics: \gazebo\odom (or \odom or \vicon_odom), \cbf_availability, \nom_policy
    - published topics: \cmd_vel, \safety_value

    '''

    '''Constructor'''
    def __init__(self):

        # name of node for ROS2
        super().__init__('safety_filter')

        '''Defining Node Attributes'''

        # define experiment parameters based on config file global variables
        self.grid_resolution = GRID_RESOLUTION
        self.state_domain = hj.sets.Box(lo=GRID_LOWER, hi=GRID_UPPER)
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(self.state_domain, self.grid_resolution, periodic_dims=PERIODIC_DIMENSIONS)

        # Required to give an arbitrary dt to the dynamics object
        self.dyn = DYNAMICS

        ''' Defining parameters for the CBF-QP Solver '''

        # Class K function alpha
        # lambda function: alpha, takes z (in this case the cbf) as an argument, and returns gamma times that argument
        # gamma is the discount factor (this dictates how fast the system will approach the edge of the safe set)
        self.alpha = lambda z: GAMMA * z

        # Control Space Constraints
        self.umin = U_MIN  # 1x2 array that defines the minimum values the linear and angular velocity can take 
        self.umax = U_MAX  # same as above line but for maximum

        # Active Set Invariance Filter (ASIF) With an initial CBF
        ###############################################

        if USE_REFINECBF is True:
            # Use the initial value function to generate the tabular CBF for safety filter
            self.tabular_cbf = refine_cbfs.TabularControlAffineCBF(self.dyn, grid=self.grid)
            self.tabular_cbf.vf_table = INITIAL_CBF
        else: 
            # load the final value function from the previous run
            self.cbf = jnp.load(CBF_FILENAME) 
            # Use the final value function to generate the tabular CBF for safety filter
                
            self.tabular_cbf = refine_cbfs.TabularControlAffineCBF(self.dyn, grid=self.grid)
            self.tabular_cbf.vf_table = np.array(self.cbf[-1])

        # ASIF declaration
        self.diffdrive_asif = ControlAffineASIF(self.dyn, self.tabular_cbf, alpha=self.alpha, umin=self.umin, umax=self.umax)

        # State and Control Variables to prevent unused variable errors
        self.state = INITIAL_STATE
        self.nominal_policy = np.array([U_MIN[0], 0])   # initial nominal policy (used to prevent errors when nominal policy table is not used if command velocity publisher is called before the nominal policy is heard)
        # self.corrective_control = np.array([0, 0]) # initial corrective control

        # quality of service profile for subscriber and publisher, provides buffer for messages
        # a depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=SAFETY_FILTER_QOS_DEPTH)

        # control publisher
        self.cmd_vel_publisher_ = self.create_publisher(
            Twist, 
            'cmd_vel',
            qos)
        
        # safety value publisher
        self.safety_value_publisher_ = self.create_publisher(
            Float32, 
            'safety_value',
            qos)

        # callback timer (how long to wait before running callback functions)
        timer_period = SAFETY_FILTER_TIMER_SECONDS
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # cbf flag subscriber
        # NOTE: If this proves to be too slow for denser grids and higher dim state spaces, replace this with float64 message of the matrix? 
        self.cbf_avail_sub = self.create_subscription(
            Bool,
            'cbf_availability',
            self.cbf_sub_callback,
            qos)

        # STATE FEEDBACK

        # create state feedback subscriber callback function
        create_sub_callback(self.__class__, STATE_FEEDBACK_TOPIC)

        # configure the state subscriber based on the state feedback source (simulation, real turtlebot3 burger, or vicon)
        self.state_sub = configure_state_feedback_subscriber(self, qos, STATE_FEEDBACK_TOPIC)
        
        # nominal policy subscriber
        self.nom_policy_sub = self.create_subscription(
            Twist,
            'nom_policy',
            self.nom_policy_sub_callback,
            qos)

        # prevent unused variable warnings
        self.cbf_avail_sub    
        self.state_sub
        self.nom_policy_sub

        # Data saving and visualization object
        self.data_logger = ParameterStorage()
    
    def publish_safety_value(self):
            
        # Safety Value at Current State (Positive = Safe, Negative/Zero = Unsafe)
        self.safety_value = self.grid.interpolate(self.tabular_cbf.vf_table, self.state)

        # Formulate the message to be published
        msg = Float32()
        msg.data = float(self.safety_value)

        # publish the safety value
        self.safety_value_publisher_.publish(msg)
        self.get_logger().info('Publishing safety value: "%s"' % msg.data)

    '''Main Loop'''
    def timer_callback(self):

        # This callback node needs to be AS FAST AS POSSIBLE. Input delays between control inputs can lead to unsafe behavior.

        # time node
        ctrl_start_time = time.time()
      
        # Compute Safety Value and Publish
        ############################################

        self.publish_safety_value()

        # CBF-QP
        ############################################
        
        # Time How Long It Takes To Solve QP
        start_time = time.time()

        # reshape to proper dimensions
        nominal_policy = np.reshape(self.nominal_policy, (1, self.dyn.control_dims))

        # Solve the QP for the optimal control input
        control_input = self.diffdrive_asif(self.state, 0.0, nominal_policy)

        print("CBF-QP solved in %s seconds" % (time.time() - start_time))

        print("Filtered Control Input:", control_input)

        # If the QP solver would return None, record the failure
        if control_input[0].any() == None:
            print("QP solver failed: Returned None")

            # Use logger to log the failure to a file for later analysis
            logging.basicConfig(filename='qp_failure.log', level=logging.DEBUG)
            logging.debug('Time of Occurence: %s' % time.time())
            logging.debug('QP solver failed: Returned None')
            logging.debug('State: "%s"' % self.state)
            logging.debug('Safety Value: "%s"' % self.safety_value)
            logging.debug('Nominal Policy: "%s"' % self.nominal_policy)

            # overwrite control for this time step to prevent program from crashing
            control_input = np.array([[U_MIN[0], 0.0]])

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

        # publish the control input
        self.cmd_vel_publisher_.publish(msg)
        self.get_logger().info('Publishing optimal safe control input over topic /cmd_vel')

        print("Time to run safety filter: %s seconds" % (time.time() - ctrl_start_time))

        # Save visualization data
        ############################################
        self.data_logger.append(x=self.state[0], y=self.state[1], theta=self.state[2], safety_value=self.safety_value, v=control_input[0,0], omega=control_input[0,1], v_nom=self.nominal_policy[0], omega_nom=self.nominal_policy[1])
          
    ##################################################
    ################### SUBSCRIBERS ##################
    ##################################################

    # Boolean flag availability subscription
    def cbf_sub_callback(self, msg):

        if USE_REFINECBF is True:

            if msg.data is True:

                self.get_logger().info('New CBF is available, loading new CBF')

                # load new tabular cbf
                self.cbf = jnp.load('./log/cbf.npy')

                # Update the tabular cbf
                self.tabular_cbf.vf_table = np.array(self.cbf)

                # Assign the tabular cbf to the diffdrive asif
                self.diffdrive_asif.cbf = self.tabular_cbf

        else:
            # skip the CBF update
            pass

    # Nominal Policy subscription
    def nom_policy_sub_callback(self, msg):

        # Message to terminal
        self.get_logger().info('Received new high level controller command.')

        self.nominal_policy = np.array([msg.linear.x, msg.angular.z])        
        
        print("Current Nominal Policy: ", self.nominal_policy)

def main():

    # initialize the node
    rclpy.init()    
    safety_filter = SafetyFilter()

    try:
        safety_filter.get_logger().info("Starting saftey filter node, shut down with CTRL+C")
        rclpy.spin(safety_filter)

    except KeyboardInterrupt:
        safety_filter.get_logger().info('Keyboard interrupt, shutting down.\n')

        # shut down motors
        msg = create_shutdown_message()
        safety_filter.cmd_vel_publisher_.publish(msg)

        # save data to file
        safety_filter.data_logger.save_data(DATA_FILENAME)

    finally:

        # shut down motors
        msg = create_shutdown_message()
        safety_filter.cmd_vel_publisher_.publish(msg)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
    safety_filter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()