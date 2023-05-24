# dynamic programming node for refine cbf experiment

import sys; sys.version

# TODO: REMOVE THESE LINES
import sys
sys.path.insert(0, '/home/nate/turtwig_ws/src/refine_cbf/refine_cbf') # for experiment_utils.py

import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool
import refine_cbfs
from refine_cbfs.dynamics import HJControlAffineDynamics
from cbf_opt import ControlAffineDynamics, ControlAffineCBF
import hj_reachability as hj
import jax.numpy as jnp
from config import *
import matplotlib.pyplot as plt
from rclpy.qos import QoSProfile
import time

class DynamicProgramming(Node):
        
    def __init__(self):

        # name of node for ROS2
        super().__init__('refine_cbf')

        # See config.py file for global variables definitions
        # defining experiment parameters
        self.grid_resolution = GRID_RESOLUTION # grid resolution for hj reachability
        self.state_domain = hj.sets.Box(lo=GRID_LOWER, hi=GRID_UPPER) # defining the state_domain
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(self.state_domain, self.grid_resolution, periodic_dims=PERIODIC_DIMENSIONS) # defining the grid object for hj reachability
        self.obst_padding = OBSTACLE_PADDING # Minkowski sum for padding
        self.obstacles = OBSTACLES # initial obstacles for the experiment from config.py file
        self.constraint_set = define_constraint_set(self.obstacles, self.obst_padding) # defining the constraint set l(x) terminal cost
        self.obstacle = hj.utils.multivmap(self.constraint_set, jnp.arange(self.grid.ndim))(self.grid.states) # defining the obstacle for the Hamilton Jacobi Reachability package
        self.umin = U_MIN # 1x2 array that defines the minimum values the linear and angular velocity can take
        self.umax = U_MAX # 1x2 array that defines the maximum values the linear and angular velocity can take
        self.dyn = DYNAMICS # instatiate an object called dyn based on the DiffDriveDynamics class
        self.dyn_jnp = DYNAMICS_JAX_NUMPY # instatiate an objected called dyn_jnp based on the DiffDriveJNPDynamics class
        self.dyn_hjr = DYNAMICS_HAMILTON_JACOBI_REACHABILITY # instatiate the dynamics for hjr using jax numpy dynamics object
        #self.dyn_hjr = DYNAMICS_HAMILTON_JACOBI_REACHABILITY_WITH_DISTURBANCE
        self.diffdrive_cbf = CBF # instatiate a diffdrive_cbf object with the Differential Drive dynamics object
        self.diffdrive_cbf = DiffDriveCBF(DYNAMICS, {"center": CENTER_CBF, "r": RADIUS_CBF, "scalar": SCALAR}, test=False) # instatiate a diffdrive_cbf object with the Differential Drive dynamics object
        self.diffdrive_tabular_cbf = refine_cbfs.TabularControlAffineCBF(self.dyn, dict(), grid=self.grid) # tabularize the cbf so that value can be calculated at each grid point
        self.diffdrive_tabular_cbf.tabularize_cbf(self.diffdrive_cbf) # tabularize the cbf so that value can be calculated at each grid point
        self.brt_fct = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))  # Backwards reachable TUBE!
        self.solver_settings = hj.SolverSettings.with_accuracy("high", value_postprocessor=self.brt_fct(self.obstacle)) # set the solver settings for hj reachability package to construct the backward reachable tube from the obstacle
        self.init_value = self.diffdrive_tabular_cbf.vf_table # initial CBF
        self.dt = TIME_STEP # time step for an HJ iteration 

        # DEBUG PLOT
        fig, ax = plt.subplots()
        ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.obstacle[..., 0], levels=[0], colors='k')
        ax.contourf(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.diffdrive_tabular_cbf.vf_table[..., 0])
        cbar = fig.colorbar(ax.contourf(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.diffdrive_tabular_cbf.vf_table[..., 0]))

        
        self.cbf_publisher_ = self.create_publisher(Bool, 'cbf_availability', 10)
        
        timer_period = 0.001  # delay between starting iterations [seconds]

        self.timer = self.create_timer(timer_period, self.timer_callback) # calls the timer_callback function every timer_period
        self.cbf_available = False

        self.counter = 0 # counter for the number of iterations of the dynamic programming loop
        self.cbf = self.init_value
        self.time = 0.0
        self.target_time = -self.dt

        # a depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=10)

    ##################################################
    ################### PUBLISHERS ###################
    ##################################################

    def publish_cbf(self):

        # set cbf flag to true
        self.cbf_available = True

        # publish boolean value, indicating a new cbf is available
        if self.cbf_available == True:

            msg = Bool()
            msg.data = True
            self.cbf_publisher_.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)
            self.cbf_available = False
    
    '''MAIN LOOP'''
    def timer_callback(self):
        
        # ********* Dynamic Programming *********
        # Use hj.step() to update the cbf and the gradient of the cbf

        self.counter += 1

        # introduce a new constraint set at specified iteration
        if self.counter == NEW_OBSTACLE_ITERATION:

            print("New obstacle introduced at iteration: ", self.counter)

            # new dictionary of obstacles
            obstacles = OBSTACLES_2

            # redifine the constraint set
            self.constraint_set = define_constraint_set(obstacles, self.obst_padding)
            # redefine the obstacle
            self.obstacle = hj.utils.multivmap(self.constraint_set, jnp.arange(self.grid.ndim))(self.grid.states)
            # redefine the brt function
            brt_fct = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))  # Backwards reachable TUBE!
            # redefine the solver settings
            self.solver_settings = hj.SolverSettings.with_accuracy("high", value_postprocessor=brt_fct(self.obstacle))

        # Refine the CBF
        # compute new iteration of value function, warmstarted using the prior

        # record time of taking a step
        start_time = time.time()

        # Iterating the CBVF using HJ Reachability using prior CBVF to warmstart the Dynamic Programming
        self.cbf = hj.step(self.solver_settings, self.dyn_hjr, self.grid, self.time, self.cbf, self.target_time, progress_bar=True)

        # record time of taking a step
        end_time = time.time()

        print("Time to take a step: ", end_time - start_time)

        save_float_to_file(end_time - start_time, './log/time_to_take_a_step.txt')

        self.time -= self.dt
        self.target_time -= self.dt        

        # save the value function
        print("Saving value function...")
        # save the current cbf to a file
        jnp.save('./log/cbf.npy', self.cbf)

        # publish boolean value indicating new cbf available
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

