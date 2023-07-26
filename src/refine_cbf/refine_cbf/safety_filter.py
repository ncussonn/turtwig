#!/usr/bin/env python3

"""Subscribes to Boolean Publisher and State Publisher and Publishes Twist Message."""

from rclpy.node import Node
from refine_cbf.config import *

class SafetyFilter(Node):
    """Safety Filter Node.

    Subscribed topics: \gazebo\odom (or \odom or \vicon_odom), \cbf_availability, \nom_policy
    Published topics: \cmd_vel, \safety_value
    """

    def __init__(self):
        """Constructor."""
        super().__init__('safety_filter')  # Name of the node for ROS2

        # Defining Node Attributes
        self.grid_resolution = GRID_RESOLUTION
        self.state_domain = hj.sets.Box(lo=GRID_LOWER, hi=GRID_UPPER)
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            self.state_domain, self.grid_resolution, periodic_dims=PERIODIC_DIMENSIONS
        )

        # Required to give an arbitrary dt to the dynamics object
        self.dyn = DYNAMICS

        # Defining parameters for the CBF-QP Solver
        self.alpha = lambda z: GAMMA * z
        self.umin = U_MIN  # 1x2 array defining the minimum values the linear and angular velocity can take
        self.umax = U_MAX  # 1x2 array defining the maximum values the linear and angular velocity can take

        # Active Set Invariance Filter (ASIF) With an initial CBF
        if USE_REFINECBF:
            self.tabular_cbf = refine_cbfs.TabularControlAffineCBF(self.dyn, grid=self.grid)
            self.tabular_cbf.vf_table = INITIAL_CBF
        else:
            self.cbf = jnp.load(CBF_FILENAME)
            self.tabular_cbf = refine_cbfs.TabularControlAffineCBF(self.dyn, grid=self.grid)
            self.tabular_cbf.vf_table = np.array(self.cbf[-1])

        self.diffdrive_asif = ControlAffineASIF(
            self.dyn, self.tabular_cbf, alpha=self.alpha, umin=self.umin, umax=self.umax
        )

        self.state = INITIAL_STATE
        self.nominal_policy = np.array([U_MIN[0], 0])  # Initial nominal policy

        # Quality of service profile for subscriber and publisher
        qos = QoSProfile(depth=SAFETY_FILTER_QOS_DEPTH)

        # Control publisher
        self.cmd_vel_publisher_ = self.create_publisher(Twist, 'cmd_vel', qos)

        # Safety value publisher
        self.safety_value_publisher_ = self.create_publisher(Float32, 'safety_value', qos)

        # Callback timer
        timer_period = SAFETY_FILTER_TIMER_SECONDS
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # CBF flag subscriber
        self.cbf_avail_sub = self.create_subscription(
            Bool, 'cbf_availability', self.cbf_sub_callback, qos
        )

        # State feedback subscriber
        create_sub_callback(self.__class__, STATE_FEEDBACK_TOPIC)

        # Configure the state subscriber based on the state feedback source
        self.state_sub = configure_state_feedback_subscriber(self, qos, STATE_FEEDBACK_TOPIC)

        # Nominal policy subscriber
        self.nom_policy_sub = self.create_subscription(
            Twist, 'nom_policy', self.nom_policy_sub_callback, qos
        )

        # Prevent unused variable warnings
        self.cbf_avail_sub
        self.state_sub
        self.nom_policy_sub

        # Data saving and visualization object
        self.data_logger = ParameterStorage()

    def publish_safety_value(self):
        """Publish the Safety Value at Current State."""
        self.safety_value = self.grid.interpolate(self.tabular_cbf.vf_table, self.state)
        msg = Float32()
        msg.data = float(self.safety_value)
        self.safety_value_publisher_.publish(msg)
        self.get_logger().info('Publishing safety value: "%s"' % msg.data)

    def timer_callback(self):
        """Main Loop."""
        ctrl_start_time = time.time()  # Time node

        # Compute Safety Value and Publish
        self.publish_safety_value()

        # CBF-QP
        start_time = time.time()  # Time how long it takes to solve QP

        nominal_policy = np.reshape(self.nominal_policy, (1, self.dyn.control_dims))
        control_input = self.diffdrive_asif(self.state, 0.0, nominal_policy)

        print("CBF-QP solved in %s seconds" % (time.time() - start_time))
        print("Filtered Control Input:", control_input)

        # If the QP solver would return None, record the failure
        if control_input[0].any() is None:
            print("QP solver failed: Returned None")
            logging.basicConfig(filename='qp_failure.log', level=logging.DEBUG)
            logging.debug('Time of Occurrence: %s', time.time())
            logging.debug('QP solver failed: Returned None')
            logging.debug('State: "%s"', self.state)
            logging.debug('Safety Value: "%s"', self.safety_value)
            logging.debug('Nominal Policy: "%s"', self.nominal_policy)
            control_input = np.array([[U_MIN[0], 0.0]])  # Overwrite control to prevent crashes

        # Publish the Optimal Control Input
        msg = Twist()
        msg.linear.x = float(control_input[0, 0])  # Linear velocity
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(control_input[0, 1])  # Angular velocity

        self.cmd_vel_publisher_.publish(msg)
        self.get_logger().info('Publishing optimal safe control input over topic /cmd_vel')

        print("Time to run safety filter: %s seconds" % (time.time() - ctrl_start_time))

        # Save visualization data
        self.data_logger.append(
            x=self.state[0],
            y=self.state[1],
            theta=self.state[2],
            safety_value=self.safety_value,
            v=control_input[0, 0],
            omega=control_input[0, 1],
            v_nom=self.nominal_policy[0],
            omega_nom=self.nominal_policy[1]
        )

    # Boolean flag availability subscription
    def cbf_sub_callback(self, msg):
        if USE_REFINECBF:
            if msg.data:
                self.get_logger().info('New CBF is available, loading new CBF')
                self.cbf = jnp.load('./log/cbf.npy')
                self.tabular_cbf.vf_table = np.array(self.cbf)
                self.diffdrive_asif.cbf = self.tabular_cbf
        else:
            # Skip the CBF update
            pass

    # Nominal Policy subscription
    def nom_policy_sub_callback(self, msg):
        """Subscriber callback for the 'nom_policy' topic."""
        # Message to terminal
        self.get_logger().info('Received new high-level controller command.')
        self.nominal_policy = np.array([msg.linear.x, msg.angular.z])

        print("Current Nominal Policy: ", self.nominal_policy)

def main():
    """Main function to start and run the node."""
    rclpy.init()
    safety_filter = SafetyFilter()

    try:
        safety_filter.get_logger().info("Starting safety filter node, shut down with CTRL+C")
        rclpy.spin(safety_filter)

    except KeyboardInterrupt:
        safety_filter.get_logger().info('Keyboard interrupt, shutting down.\n')

        # Shut down motors
        msg = create_shutdown_message()
        safety_filter.cmd_vel_publisher_.publish(msg)

        # Save data to file
        safety_filter.data_logger.save_data(DATA_FILENAME)

    finally:
        # Shut down motors
        msg = create_shutdown_message()
        safety_filter.cmd_vel_publisher_.publish(msg)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        safety_filter.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
