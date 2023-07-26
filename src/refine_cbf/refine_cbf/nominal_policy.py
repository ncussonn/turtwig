#!/usr/bin/env python3

# Nominal policy node for refineCBF

# This is an example implementation which uses a precomputed nominal policy table derived from solving a Hamilton Jacobi Reachability reachability problem offline.

from rclpy.node import Node
from refine_cbf.config import *
from refine_cbf.high_level_controller import NominalController

class NominalPolicy(Node):
    """
    Nominal Policy Node

    Subscribed topics: /gazebo/odom or /odom or /vicon_odom
    Published topics: /nom_policy
    """

    def __init__(self):
        super().__init__('nominal_policy')

        # Required to give an arbitrary dt to the dynamics object
        # TODO: rewrite dynamics object to not have these required parameters
        self.dyn = DiffDriveDynamics({"dt": 0.05}, test=False)

        # Environment Parameters
        self.grid = GRID

        # Initializing common parameters
        self.state = INITIAL_STATE  # initial state
        self.nominal_policy = np.array([0.1, 0])  # initial nominal policy (used to prevent errors when nominal policy table is not used if command velocity publisher is called before the nominal policy is heard)
        self.high_level_controller = NominalController(self)

        # Quality of service profile for subscriber and publisher, provides buffer for messages
        # A depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=NOMINAL_POLICY_QOS_DEPTH)

        # Control publisher
        self.nom_pol_publisher_ = configure_nominal_policy_publisher(self, qos, USE_UNFILTERED_POLICY)

        # Callback timer (how long to wait before running the callback function)
        timer_period = NOMINAL_POLICY_TIMER_SECONDS
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # State feedback
        # Create the state feedback subscriber callback function
        create_sub_callback(self.__class__, STATE_FEEDBACK_TOPIC)

        # Configure the state subscriber based on the state feedback source (simulation, real turtlebot3 burger, or vicon)
        self.state_sub = configure_state_feedback_subscriber(self, qos, STATE_FEEDBACK_TOPIC)

        # Prevent unused variable warnings
        self.state_sub

        # Data saving and visualization object
        self.data_logger = ParameterStorage()

    def timer_callback(self):
        """
        Timer callback function for the main loop.
        """
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
        msg.linear.x = float(self.nominal_policy[0, 0])  # linear velocity
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(self.nominal_policy[0, 1])  # angular velocity

        # Publish the control input
        self.nom_pol_publisher_.publish(msg)
        publish_message = create_nominal_policy_publishing_message(USE_UNFILTERED_POLICY)
        self.get_logger().info(publish_message)

        # Save visualization data
        ############################################
        if USE_UNFILTERED_POLICY:
            self.data_logger.append(x=self.state[0], y=self.state[1], theta=self.state[2], v_nom=self.nominal_policy[0, 0], omega_nom=self.nominal_policy[0, 1])

def main():
    # Initialize the node
    rclpy.init()
    nominal_policy = NominalPolicy()

    try:
        nominal_policy.get_logger().info("Starting saftey filter node, shut down with CTRL+C")
        rclpy.spin(nominal_policy)

    except KeyboardInterrupt:

        # Redundant shutdown command as a failsafe

        # When ctrl+c is pressed, do the following
        nominal_policy.get_logger().info('Keyboard interrupt, shutting down.\n')

        # Shutdown protocol
        msg = create_shutdown_message()
        nominal_policy.nom_pol_publisher_.publish(msg)
        if USE_UNFILTERED_POLICY:
            nominal_policy.data_logger.save_data(DATA_FILENAME)

    finally:

        # Shutdown protocol
        msg = create_shutdown_message()
        nominal_policy.nom_pol_publisher_.publish(msg)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    nominal_policy.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()