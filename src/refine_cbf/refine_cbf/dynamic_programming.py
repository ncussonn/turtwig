# dynamic programming node for refine cbf experiment

# Publishes:
#  cbf availability
#  

import sys; sys.version

# TODO: REMOVE THESE LINES
import sys
sys.path.insert(0, '/home/nate/turtwig_ws/src/refine_cbf/refine_cbf') # for experiment_utils.py

import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool
#from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import refine_cbfs
from refine_cbfs.dynamics import HJControlAffineDynamics
from cbf_opt import ControlAffineDynamics, ControlAffineCBF
import hj_reachability as hj
import jax.numpy as jnp
from config import *
import matplotlib.pyplot as plt

class DynamicProgramming(Node):

    

    # # Control Space Constraints
    # vmin = 0.1      # minimum linear velocity of turtlebot3 burger
    # vmax = 0.21     # maximum linear velocity of turtlebot3 burger
    # wmax = 2.63     # maximum angular velocity of turtlebot3 burger

    # umin = np.array([vmin, -wmax]) # 1x2 array that defines the minimum values the linear and angular velocity can take 
    # umax = np.array([vmax, wmax])  # same as above line but for maximum

    # defining the state_domain
    #state_domain = hj.sets.Box(lo=jnp.array([0., 0., -jnp.pi]), hi=jnp.array([2., 2., jnp.pi]))
    
    
    # defining the grid for hj reachability
    #grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, grid_resolution, periodic_dims=2)

    # Minkowski sum for padding
    #obst_padding = 0.11 # max radius of turtlebot3 burger is about 0.11 m

    # obstacles = {"circle": {"center": np.array([1.0, 1.0]), "radius": 0.15},
    #              "bounding_box": {"center": np.array([1.0, 1.0]),"length": np.array([2.0,2.0])},}
    #             #  "rectangle": {"center": np.array([0.5, 1.5]),"length": np.array([0.5,0.5])}}

    # # defining the constraint set
    # constraint_set = define_constraint_set(obstacles, obst_padding)

    # # defining the obstacle for the Hamilton Jacobi Reachability package
    # obstacle = hj.utils.multivmap(constraint_set, jnp.arange(grid.ndim))(grid.states)  # l(x) terminal cost

    # instantiate an object called dyn based on the DiffDriveDynamics class
    # constructor parameters: dt = 0.05, test = False
    # dyn = DiffDriveDynamics({"dt": dt}, test=False)

    # # instantiate an objected called dyn_jnp based on the DiffDriveJNPDynamics class
    # # constructor parameters: dt = 0.05, test = False
    # dyn_jnp = DiffDriveJNPDynamics({"dt": dt}, test=False)

    # # instatiate the dynamics for hjr using jax numpy dynamics object
    # dyn_hjr = HJControlAffineDynamics(dyn_jnp, control_space=hj.sets.Box(jnp.array(umin), jnp.array(umax)))

    # # instatiate a diffdrive_cbf object with the Differential Drive dynamics object
    # diffdrive_cbf = DiffDriveCBF(dyn, {"center": np.array([0.5, 0.5]), "r": 0.25, "scalar": 2.}, test=False)

    # # tabularizing the cbf so that value can be calculated at each grid point
    # diffdrive_tabular_cbf = refine_cbfs.TabularControlAffineCBF(dyn, dict(), grid=grid)
    # diffdrive_tabular_cbf.tabularize_cbf(diffdrive_cbf)

    # # DEBUG PLOT
    # fig, ax = plt.subplots()
    # ax.contour(grid.coordinate_vectors[1], grid.coordinate_vectors[0], obstacle[..., 0], levels=[0], colors='k')
    # ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0], diffdrive_tabular_cbf.vf_table[..., 0])
    # cbar = fig.colorbar(ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0], diffdrive_tabular_cbf.vf_table[..., 0]))

    # # formulate the backward reachable tube of the obstacle
    # # lambda function: obstacle is argument, another lambda function with t, x as argument
    # brt_fct = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))  # Backwards reachable TUBE!
    # # set the solver settings for hj reachability package to construct the backward reachable tube from the obstacle
    # solver_settings = hj.SolverSettings.with_accuracy("high", value_postprocessor=brt_fct(obstacle))

    # # initial value function (tabularized)
    # init_value = diffdrive_tabular_cbf.vf_table
        
    def __init__(self):

        # See config.py file for global variables definitions
        # defining experiment parameters
        self.grid_resolution = GRID_RESOLUTION
        self.state_domain = hj.sets.Box(lo=GRID_LOWER, hi=GRID_UPPER)
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(self.state_domain, self.grid_resolution, periodic_dims=PERIODIC_DIMENSIONS)
        self.obst_padding = OBSTACLE_PADDING
        self.obstacles = OBSTACLES
        self.constraint_set = define_constraint_set(self.obstacles, self.obst_padding)
        self.obstacle = hj.utils.multivmap(self.constraint_set, jnp.arange(self.grid.ndim))(self.grid.states)
        self.umin = U_MIN
        self.umax = U_MAX
        self.dyn = DYNAMICS
        self.dyn_jnp = DYNAMICS_JAX_NUMPY
        self.dyn_hjr = DYNAMICS_HAMILTON_JACOBI_REACHABILITY
        self.diffdrive_cbf = CBF
        self.diffdrive_cbf = DiffDriveCBF(DYNAMICS, {"center": CENTER_CBF, "r": RADIUS_CBF, "scalar": SCALAR}, test=False)
        self.diffdrive_tabular_cbf = refine_cbfs.TabularControlAffineCBF(self.dyn, dict(), grid=self.grid)
        self.diffdrive_tabular_cbf.tabularize_cbf(self.diffdrive_cbf) # tabularize the cbf so that value can be calculated at each grid point
        self.brt_fct = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))  # Backwards reachable TUBE!
        self.solver_settings = hj.SolverSettings.with_accuracy("high", value_postprocessor=self.brt_fct(self.obstacle))
        self.init_value = self.diffdrive_tabular_cbf.vf_table # initial CBF
        # time horizon over which to compute the BRT
        self.dt = TIME_STEP # time step for an HJ iteration 

        # DEBUG PLOT
        fig, ax = plt.subplots()
        ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.obstacle[..., 0], levels=[0], colors='k')
        ax.contourf(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.diffdrive_tabular_cbf.vf_table[..., 0])
        cbar = fig.colorbar(ax.contourf(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.diffdrive_tabular_cbf.vf_table[..., 0]))

        super().__init__('dp_node')
        self.cbf_publisher_ = self.create_publisher(Bool, 'cbf_availability', 10)
        self.obstacle_1_publisher_ = self.create_publisher(Path, 'obstacle_1', 10)
        self.obstacle_2_publisher_ = self.create_publisher(Path, 'obstacle_2', 10)
        self.obstacle_3_publisher_ = self.create_publisher(Path, 'obstacle_3', 10)
        self.obstacle_4_publisher_ = self.create_publisher(Path, 'obstacle_4', 10)
        self.obstacle_5_publisher_ = self.create_publisher(Path, 'obstacle_5', 10)
        # self.internal_obstacle_publisher_ = self.create_publisher(Path, 'obstacle_internal', 10)
        self.safe_set_1_publisher_ = self.create_publisher(Path, 'safe_set_1', 10)
        self.safe_set_2_publisher_ = self.create_publisher(Path, 'safe_set_2', 10)
        self.safe_set_3_publisher_ = self.create_publisher(Path, 'safe_set_3', 10)
        self.safe_set_4_publisher_ = self.create_publisher(Path, 'safe_set_4', 10)
        self.safe_set_5_publisher_ = self.create_publisher(Path, 'safe_set_5', 10)
        # self.internal_safe_set_publisher_ = self.create_publisher(Path, 'safe_set_internal', 10)
        self.initial_safe_set_publisher_ = self.create_publisher(Path, 'initial_safe_set', 10)

        timer_period = 0.001  # delay between starting iterations [seconds]

        self.timer = self.create_timer(timer_period, self.timer_callback) # calls the timer_callback function every timer_period
        self.cbf_available = False
        self.safe_set_vertices = []
        self.initial_safe_set_vertices = []

    def generate_pose_stamp(self, vertex, time_stamp):

        # generates a pose stamped message by using the vertices of the safe set
        pose_stamp_message = PoseStamped()

        # set the header
        pose_stamp_message.header.frame_id = "odom"
        pose_stamp_message.header.stamp = time_stamp
        
        # set the pose
        
        # set the position
        pose_stamp_message.pose.position.x = vertex[0]
        pose_stamp_message.pose.position.y = vertex[1]
        pose_stamp_message.pose.position.z = 0.0 # project onto the ground

        # set the orientation (this will be irrelevant, as we only care about the position)
        pose_stamp_message.pose.orientation.x = 0.0
        pose_stamp_message.pose.orientation.y = 0.0
        pose_stamp_message.pose.orientation.z = 0.0
        pose_stamp_message.pose.orientation.w = 1.0

        return pose_stamp_message
    
    def create_path_msg(self, points):

        # create a Path message
        msg = Path()

        # generate a time stamp for the path message
        time_stamp = self.get_clock().now().to_msg()

        # set the header
        msg.header.frame_id = "odom"
        msg.header.stamp = time_stamp        

        # set the poses
        for i in range(len(points)):
            msg.poses.append(self.generate_pose_stamp(points[i], time_stamp))

        return msg

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
    
    def publish_safe_set(self, paths):

        # # create a path message based on the current safe set vertices
        # msg = self.create_path_msg(vertices)    

        # Only supports up to 5 holes in the safe set
        for i in range(len(paths)):

            path = paths[i]
            vertices = path.vertices
            msg = self.create_path_msg(vertices)

            if i == 0:
                self.safe_set_1_publisher_.publish(msg)
            elif i == 1:
                self.safe_set_2_publisher_.publish(msg)
            elif i == 2:
                self.safe_set_3_publisher_.publish(msg)
            elif i == 3:
                self.safe_set_4_publisher_.publish(msg)
            elif i == 4:
                self.safe_set_5_publisher_.publish(msg)
            else:
                self.get_logger().info('Disjoint Obstacle Limit Reached. Unable to visualize all safe set holes.')

            self.get_logger().info('Publishing: Safe Set Path')

        # # publish the message
        # self.safe_set_publisher_.publish(msg)
        # self.get_logger().info('Publishing: Safe Set Path')

    def publish_internal_safe_set(self, vertices):

        # create a path message based on the inner contour of current safe set vertices
        msg = self.create_path_msg(vertices)    

        # publish the message
        self.internal_safe_set_publisher_.publish(msg)
        self.get_logger().info('Publishing: Safe Set Path (Internal)')

    def publish_obstacles(self, paths):
     
        # clear the unused obstacle publishers if there are any

        # create an empty path message
        vertices = []
        msg = self.create_path_msg(vertices)

        for i in range(5, len(paths)-1, -1):
            print(i)

            if i == 0:
                self.obstacle_1_publisher_.publish(msg)
            elif i == 1:
                self.obstacle_2_publisher_.publish(msg)
            elif i == 2:
                self.obstacle_3_publisher_.publish(msg)
            elif i == 3:
                self.obstacle_4_publisher_.publish(msg)
            elif i == 4:
                self.obstacle_5_publisher_.publish(msg)

        # publish the obstacle paths
        # Only supports up to 5 disjoint obstacles
        for i in range(len(paths)):
            print(i)

            path = paths[i]
            vertices = path.vertices
            msg = self.create_path_msg(vertices)

            if i == 0:
                self.obstacle_1_publisher_.publish(msg)
            elif i == 1:
                self.obstacle_2_publisher_.publish(msg)
            elif i == 2:
                self.obstacle_3_publisher_.publish(msg)
            elif i == 3:
                self.obstacle_4_publisher_.publish(msg)
            elif i == 4:
                self.obstacle_5_publisher_.publish(msg)
            else:
                self.get_logger().info('Disjoint Obstacle Limit Reached. Unable to visualize all obstacles.')

            self.get_logger().info('Publishing: Obstacle Path')

        # publish the message
        # self.obstacle_publisher_.publish(msg)
        # self.get_logger().info('Publishing: Obstacle Path')

    def publish_internal_obstacle(self, vertices):

        # create a path message based on obstacle vertices
        msg = self.create_path_msg(vertices)

        # publish the message
        self.internal_obstacle_publisher_.publish(msg)
        self.get_logger().info('Publishing: Internal Obstacle Path')

    def publish_initial_safe_set(self, vertices):

        # create a path message based on the initial safe set vertices
        msg = self.create_path_msg(vertices) 

        # publish the message
        self.initial_safe_set_publisher_.publish(msg)
        self.get_logger().info('Publishing: Initial Safe Set Path')


    def timer_callback(self):
        
        # ********* Dynamic Programming *********
        # Use hj.step() to update the cbf and the gradient of the cbf

        self.cbf = self.init_value
        time = 0.0
        target_time = -self.dt

        print("Shape of cbf (State Space Grid Size): ", self.cbf.shape)

        # initiate figure
        fig, ax = plt.subplots(1, 1, figsize=(18,18 ))

        # Initial contour plot of the 0 superlevel set
        safe_set_contour = ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.cbf[..., -1], levels=[0], colors='k')

        # retrieve vertices of the contour for Rviz visualization
        self.initial_safe_set_vertices = safe_set_contour.collections[0].get_paths()[0].vertices

        # create contour of obstacle
        obst_contour = ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.obstacle[..., 0], levels=[0], colors='r')

        # retrieve the vertices of the obstacle contour
        self.obst_vertices = obst_contour.collections[0].get_paths()[0].vertices

        # iterate bool
        iterate = True

        counter = 0

        # # publish the initial safe set and obstacle
        self.publish_initial_safe_set(self.initial_safe_set_vertices)
        # self.publish_obstacles(self.obst_vertices) 

        self.publish_obstacles(obst_contour.collections[0].get_paths()) 

        # infinite while loop, simulates infinite time horizon
        while iterate is True:
            
            counter += 1

            # introduce a new obstacle
            if counter == 50:

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

            # compute new iteration of value function, warmstarted using the prior
            self.cbf = hj.step(self.solver_settings, self.dyn_hjr, self.grid, time, self.cbf, target_time, progress_bar=True)
            time -= self.dt
            target_time -= self.dt

            # plot the contour of the 0 level set of the cbf (the safe set) at particular theta slice
            theta_slice = 11
            safe_set_contour = ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.cbf[..., theta_slice], levels=[0], colors='k')   
            # create contour of obstacle
            obst_contour = ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.obstacle[..., 0], levels=[0], colors='r')

            # contour paths
            safe_set_paths = safe_set_contour.collections[0].get_paths()
            obstacle_paths = obst_contour.collections[0].get_paths()

            # obtain vertices of the contour for Rviz visualization
            self.safe_set_vertices = safe_set_paths[0].vertices

            # Publish contours as paths for Rviz visualization
            self.publish_obstacles(obstacle_paths) 
            self.publish_safe_set(safe_set_paths) 

            # in order to show the safe set contour which envelops an obstacle, we must extract the vertices of the second contour
            # the length of the paths list will exceed 1 if the safe set contour envelops an obstacle
            # if len(paths_safe_set) > 1:
            #     internal_path = safe_set_contour.collections[0].get_paths()[1]
            #     internal_vertices = internal_path.vertices
            #     self.publish_internal_safe_set(internal_vertices)

            # # to show internal obstacles contour, we must extract the vertices of the additional contours
            # if len(paths_obstacle) > 1:
            #     internal_path = obst_contour.collections[0].get_paths()[1]
            #     internal_vertices = internal_path.vertices
            #     self.publish_internal_obstacle(internal_vertices)

            # save the current cbf to a file
            jnp.save('./log/cbf.npy', self.cbf)

            # publish boolean value indicating new cbf available
            self.publish_cbf()

            # save the value function
            print("Saving value function...")

            # # prompt user if they want to generate another cbf
            # iterate = input("Generate another cbf? (y/n): ")
            # #iterate = 'y'

            # if iterate == 'y':
            #     iterate = True
            # else:
            #     iterate = False

        # show figure
        plt.show()
        # stop node
        rclpy.shutdown()

def main():

    print("Starting dynamic programming node...")

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

