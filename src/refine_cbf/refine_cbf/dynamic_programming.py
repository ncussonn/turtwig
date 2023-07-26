#!/usr/bin/env python3

# Dynamic programming node for refine cbf experiment

from rclpy.node import Node
from refine_cbf.config import *

class DynamicProgramming(Node):
    """
    Dynamic Programming Node for refineCBF

    Subscribed topics: None
    Published topics: /cbf_availability (std_msgs/Bool)
    """
    def __init__(self):
        super().__init__('refine_cbf')

        self.grid = GRID
        self.obst_padding = OBSTACLE_PADDING  # Minkowski sum for padding
        self.obstacles = OBSTACLE_LIST[0]  # initial obstacles for the experiment from config.py file
        self.constraint_set = define_constraint_function(self.obstacles, self.obst_padding)  # defining the constraint set l(x) terminal cost
        self.obstacle_hjr = hj.utils.multivmap(self.constraint_set, jnp.arange(self.grid.ndim))(self.grid.states)  # defining the obstacle for the Hamilton Jacobi Reachability package
        self.dyn = DYNAMICS  # instantiate an object called dyn based on the DiffDriveDynamics class
        self.dyn_hjr = DYNAMICS_HAMILTON_JACOBI_REACHABILITY  # instantiate the dynamics for hjr using jax numpy dynamics object
        self.diffdrive_cbf = CBF  # instantiate a diffdrive_cbf object with the Differential Drive dynamics object
        self.diffdrive_tabular_cbf = refine_cbfs.TabularControlAffineCBF(self.dyn, dict(), grid=self.grid)  # tabularize the cbf so that value can be calculated at each grid point
        self.diffdrive_tabular_cbf.tabularize_cbf(self.diffdrive_cbf)  # tabularize the cbf so that value can be calculated at each grid point
        self.brt_fct = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))  # Backwards reachable TUBE!
        self.solver_settings = hj.SolverSettings.with_accuracy("high", value_postprocessor=self.brt_fct(self.obstacle_hjr))  # set the solver settings for hj reachability package to construct the backward reachable tube from the obstacle
        self.init_value = self.diffdrive_tabular_cbf.vf_table  # initial CBF
        self.dt = TIME_STEP  # time step for an HJ iteration

        self.cbf_publisher_ = self.create_publisher(Bool, 'cbf_availability', 10)

        timer_period = 0.001  # delay between starting iterations [seconds]

        self.timer = self.create_timer(timer_period, self.timer_callback)  # calls the timer_callback function every timer_period
        self.cbf_available = False

        self.iteration = 0  # counter for the number of iterations of the dynamic programming loop
        self.cbf = self.init_value
        self.time = 0.0
        self.target_time = -self.dt

        # a depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=10)

    def publish_cbf(self):
        """
        Publish the cbf availability message.
        """
        # set cbf flag to true
        self.cbf_available = True

        # publish boolean value, indicating a new cbf is available
        if self.cbf_available:
            msg = Bool()
            msg.data = True
            self.cbf_publisher_.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)
            self.cbf_available = False

    def timer_callback(self):
        """
        Timer callback function for the main loop.
        """
        # ********* Dynamic Programming *********
        # Use hj.step() to update the cbf and the gradient of the cbf

        # keep track of the number of iterations of the dynamic programming loop
        self.iteration += 1

        # check if a new obstacle should be introduced at the current iteration, and if so, update the CBF using it
        # NOTE: Updating the constraint set / solver settings takes a while for larger state spaces even at 3 dimensions.
        # This appears to result from the HJR package and interferes with the dynamic programming loop.
        # It would require rewriting part of the HJR package to solve the issue - so swift updates to the obstacle set may not be possible.
        # One way of circumventing this would be to proactively create the solver settings for each different obstacle set offline,
        # and introduce them during the respective iteration.
        # Alternatively, another node could be made that dynamic programming subscribes to, which gets the newest solver settings that contain
        # the new obstacle set.
        # If the obstacle set is not known a priori, and is updated rapidly online, then this implementation will likely pose some
        # issues due to the HJR package.
        introduce_obstacle(self, OBSTACLE_LIST, OBSTACLE_ITERATION_LIST)

        # record time of taking a step
        start_time = time.time()

        # Iterating the CBVF using HJ Reachability using the prior CBVF to warmstart the Dynamic Programming
        self.cbf = hj.step(self.solver_settings, self.dyn_hjr, self.grid, self.time, self.cbf, self.target_time, progress_bar=True)

        # record time of taking a step
        end_time = time.time()

        print("Time for iteration: ", end_time - start_time)

        self.time -= self.dt
        self.target_time -= self.dt

        # save the value function
        print("Saving value function...")
        # save the current cbf to a file
        np.save('./log/cbf.npy', self.cbf)

        # publish boolean value indicating a new cbf is available
        self.publish_cbf()

def main():
    rclpy.init()
    dynamic_programming = DynamicProgramming()

    try:
        dynamic_programming.get_logger().info("Starting dynamic programming node, shut down with CTRL+C")
        rclpy.spin(dynamic_programming)

    except KeyboardInterrupt:
        dynamic_programming.get_logger().info('Keyboard interrupt, shutting down.\n')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dynamic_programming.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
