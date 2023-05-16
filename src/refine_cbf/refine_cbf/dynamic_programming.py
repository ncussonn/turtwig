# dynamic programming node for refine cbf experiment

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


from rclpy.qos import QoSProfile

class DynamicProgramming(Node):
        
    def __init__(self):

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

        self.counter = 0
        self.cbf = self.init_value
        self.time = 0.0
        self.target_time = -self.dt

        # a depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=10)

        # simulation state subscription
        self.state_sub = self.create_subscription(
            Odometry,
            'gazebo/odom',
            self.state_sub_callback,
            qos)
    
        self.state = INITIAL_STATE

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

        # create an empty path message
        vertices = []
        msg = self.create_path_msg(vertices)

        for i in range(5, len(paths)-1, -1):
            
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

    def publish_obstacles(self, paths):
     
        # clear the unused obstacle publishers if there are any

        # create an empty path message
        vertices = []
        msg = self.create_path_msg(vertices)

        for i in range(5, len(paths)-1, -1):

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

    def publish_initial_safe_set(self, vertices):

        # create a path message based on the initial safe set vertices
        msg = self.create_path_msg(vertices) 

        # publish the message
        self.initial_safe_set_publisher_.publish(msg)
        self.get_logger().info('Publishing: Initial Safe Set Path')

    '''MAIN LOOP'''
    def timer_callback(self):
        
        # ********* Dynamic Programming *********
        # Use hj.step() to update the cbf and the gradient of the cbf

        #print("Shape of cbf (State Space Grid Size): ", self.cbf.shape)

        # Plot the nearest theta slice to the current heading angle
        # theta_slice = self.grid.nearest_index(self.state)[2]
        # print("Theta Slice: ", theta_slice)

        # if self.counter == 0:

        #     # initiate a figure for matplotlib to get the contour plot of the cbf
        #     fig, self.ax = plt.subplots(1, 1, figsize=(18,18 ))

        #     # Initial contour plot of the 0 superlevel set
        #     #safe_set_contour = ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.cbf[..., -1], levels=[0], colors='k')
        #     safe_set_contour = self.ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.cbf[..., theta_slice], levels=[0], colors='k')

        #     # retrieve vertices of the contour for Rviz visualization
        #     self.initial_safe_set_vertices = safe_set_contour.collections[0].get_paths()[0].vertices

        #     # create contour of obstacle
        #     self.obst_contour = self.ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.obstacle[..., 0], levels=[0], colors='r')

        #     # retrieve the vertices of the obstacle contour
        #     self.obst_vertices = self.obst_contour.collections[0].get_paths()[0].vertices

        #     # iterate bool
        #     # iterate = True

        #     # # publish the initial safe set and obstacle
        #     #self.publish_initial_safe_set(self.initial_safe_set_vertices)
        #     # self.publish_obstacles(self.obst_vertices) 

        #     #self.publish_obstacles(self.obst_contour.collections[0].get_paths()) 

        self.counter += 1

        # introduce a new constraint set
        if self.counter == 30:

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
        self.cbf = hj.step(self.solver_settings, self.dyn_hjr, self.grid, self.time, self.cbf, self.target_time, progress_bar=True)
        self.time -= self.dt
        self.target_time -= self.dt        

        # # plot the contour of the 0 level set of the cbf (the safe set) at particular theta slice
        # safe_set_contour = self.ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.cbf[..., theta_slice], levels=[0], colors='k')   
        # # create contour of obstacle
        # self.obst_contour = self.ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.obstacle[..., 0], levels=[0], colors='r')

        # # contour paths
        # safe_set_paths = safe_set_contour.collections[0].get_paths()
        # obstacle_paths = self.obst_contour.collections[0].get_paths()

        # # obtain vertices of the contour for Rviz visualization
        # self.safe_set_vertices = safe_set_paths[0].vertices

        # Publish contours as paths for Rviz visualization
        #self.publish_obstacles(obstacle_paths) 
        #self.publish_safe_set(safe_set_paths) 

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
        #plt.show()
        # stop node
        #rclpy.shutdown()

    # Simulation State Subscription
    def state_sub_callback(self, msg):

        # Message to terminal
        self.get_logger().info('Received new simulation tate.')

        # convert quaternion to euler angle
        (roll, pitch, yaw) = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        self.state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        
        print("Current Simulation State: ", self.state)

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

