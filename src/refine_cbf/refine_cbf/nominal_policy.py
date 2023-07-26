#!/usr/bin/env python3

# Nominal policy node for refineCBF

# This is an example implementation which uses a precomputed nominal policy table derived from solving a Hamilton Jacobi Reachability reachability problem offline.

from rclpy.node import Node
from refine_cbf.config import *
from refine_cbf.high_level_controller import NominalController

class NominalPolicy(Node):

    '''
    Noimnal Policy Node

    - subscribed topics: \gazebo\odom or \odom or \vicon_odom
    - published topics:  \nom_policy

    '''

    '''Constructor'''
    def __init__(self):

        # name of node for ROS2
        super().__init__('nominal_policy')

        '''Defining Node Attributes'''

        # Required to give an arbitrary dt to the dynamics object
        # TODO: rewrite dynamics object to not have these required parameters
        self.dyn = DiffDriveDynamics({"dt": 0.05}, test=False)

        '''Environment Parameters'''

        self.grid = GRID

        # initializing common parameters
        self.state = INITIAL_STATE  # initial state
        self.nominal_policy = np.array([0.1, 0])   # initial nominal policy (used to prevent errors when nominal policy table is not used if command velocity publisher is called before the nominal policy is heard)
        self.high_level_controller = NominalController(self)

        # quality of service profile for subscriber and publisher, provides buffer for messages
        # a depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=NOMINAL_POLICY_QOS_DEPTH)

        # control publisher
        self.nom_pol_publisher_ = configure_nominal_policy_publisher(self, qos, USE_UNFILTERED_POLICY)
        
        # callback timer (how long to wait before running callback function)
        timer_period = NOMINAL_POLICY_TIMER_SECONDS
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # STATE FEEDBACK

        # create the state feedback subscriber callback function
        create_sub_callback(self.__class__, STATE_FEEDBACK_TOPIC)

        # configure the state subscriber based on the state feedback source (simulation, real turtlebot3 burger, or vicon)
        self.state_sub = configure_state_feedback_subscriber(self, qos, STATE_FEEDBACK_TOPIC)
    
        # prevent unused variable warnings    
        self.state_sub

        # Data saving and visualization object
        self.data_logger = ParameterStorage()

    '''Main Loop'''
    def timer_callback(self):
        
        # Compute Nominal Control
        ############################################

        # Offline, a nominal policy table was computed for every grid point in the state space.

        # Get value of the nominal policy at the current state using a precomputed nominal policy table (nominal policy will be interpolated)

        self.nominal_policy = self.high_level_controller.compute_nominal_control(self)

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

def main():

    # initialize the node
    rclpy.init()    
    nominal_policy = NominalPolicy()

    try:
        nominal_policy.get_logger().info("Starting saftey filter node, shut down with CTRL+C")
        rclpy.spin(nominal_policy)

    except KeyboardInterrupt:

        # redundant shutdown command as a failsafe

        # when ctrl+c is pressed, do the following
        nominal_policy.get_logger().info('Keyboard interrupt, shutting down.\n')

        # shutdown protocol
        msg = create_shutdown_message()
        nominal_policy.nom_pol_publisher_.publish(msg)
        nominal_policy.data_logger.save_data(DATA_FILENAME_NOMINAL_POLICY)

    finally:

        # shutdown protocol
        msg = create_shutdown_message()
        nominal_policy.nom_pol_publisher_.publish(msg)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    nominal_policy.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()