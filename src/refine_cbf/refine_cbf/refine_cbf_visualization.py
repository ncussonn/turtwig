#!/usr/bin/env python3

# TODO: Occasionally, this script crashes due to the loading of the new cbf happening as it is being saved to. Should try and find a workaround.

# sends visualization topics to rviz

from rclpy.node import Node
from refine_cbf.config import *

class RefineCBF_Visualization(Node):

    '''
    RVIZ Visualization Node for refineCBF

    - subscribed topics: \gazebo\odom (or \odom or \vicon_odom), \cbf_availability
    - published topics: \obstacle_1, \obstacle_2, \obstacle_3, \obstacle_4, \obstacle_5, \safe_set_1, \safe_set_2, \safe_set_3, \safe_set_4, \safe_set_5, \initial_safe_set, \goal_set

    '''
        
    def __init__(self):

        # node name in ROS2 network
        super().__init__('visualizer')

        # See config.py file for GLOBAL variables definitions
        # defining experiment parameters
        # self.grid_resolution = GRID_RESOLUTION # grid resolution for hj reachability
        # self.state_domain = hj.sets.Box(lo=GRID_LOWER, hi=GRID_UPPER) # defining the state_domain
        # self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(self.state_domain, self.grid_resolution, periodic_dims=PERIODIC_DIMENSIONS) # defining the grid object for hj reachability
        self.grid = GRID
        self.obst_padding = OBSTACLE_PADDING # Minkowski sum for padding
        self.constraint_set = define_constraint_function(OBSTACLE_LIST[0], OBSTACLE_PADDING) # defining the constraint set l(x) terminal cost
        self.obstacle = hj.utils.multivmap(self.constraint_set, jnp.arange(self.grid.ndim))(self.grid.states) # defining the obstacle for the Hamilton Jacobi Reachability package

        # a depth of 10 suffices in most cases, but this can be increased if needed
        qos = QoSProfile(depth=10)

        self.obstacle_1_publisher_ = self.create_publisher(Path, 'obstacle_1', qos)
        self.obstacle_2_publisher_ = self.create_publisher(Path, 'obstacle_2', qos)
        self.obstacle_3_publisher_ = self.create_publisher(Path, 'obstacle_3', qos)
        self.obstacle_4_publisher_ = self.create_publisher(Path, 'obstacle_4', qos)
        self.obstacle_5_publisher_ = self.create_publisher(Path, 'obstacle_5', qos)
        self.safe_set_1_publisher_ = self.create_publisher(Path, 'safe_set_1', qos)
        self.safe_set_2_publisher_ = self.create_publisher(Path, 'safe_set_2', qos)
        self.safe_set_3_publisher_ = self.create_publisher(Path, 'safe_set_3', qos)
        self.safe_set_4_publisher_ = self.create_publisher(Path, 'safe_set_4', qos)
        self.safe_set_5_publisher_ = self.create_publisher(Path, 'safe_set_5', qos)
        self.initial_safe_set_publisher_ = self.create_publisher(Path, 'initial_safe_set', qos)
        self.goal_set_publisher_ = self.create_publisher(Path, 'goal_set', qos)

        timer_period = 0.005  # delay between starting iterations [seconds]

        self.timer = self.create_timer(timer_period, self.timer_callback) # calls the timer_callback function every timer_period
        self.cbf_available = False # set cbf availability to false initially
        
        self.safe_set_vertices = []
        self.initial_safe_set_vertices = []

        self.iteration = 0
        self.cbf = INITIAL_CBF

        # cbf to save
        self.cbf_to_save = {self.iteration:INITIAL_CBF}
        
        # create the state feedback subscriber callback function
        create_sub_callback(self.__class__, STATE_FEEDBACK_TOPIC)

        # initial state
        self.state = INITIAL_STATE

        self.state_sub = configure_state_feedback_subscriber(self, qos, topic_string=STATE_FEEDBACK_TOPIC)

        # cbf flag subscriber
        self.cbf_avail_sub = self.create_subscription(
            Bool,
            'cbf_availability',
            self.cbf_sub_callback,
            qos)

    # Boolean flag availability subscription
    # NOTE: If this proves to be too slow for denser grids and higher dim state spaces, replace this with float64 message of the matrix? 
    def cbf_sub_callback(self, bool_msg):

        self.get_logger().info('New CBF Available: "%s"' % bool_msg.data)

        if bool_msg.data is True:
            
            # increment the counter
            self.iteration+=1

            # Halt the robot to prevent unsafe behavior while the CBF is updated
            self.get_logger().info('New CBF received, loading new CBF')

            # load new tabular cbf
            self.cbf = jnp.load('./log/cbf.npy')

            # update the cbf to save
            self.cbf_to_save[self.iteration] = self.cbf

    def generate_pose_stamp(self, vertex, time_stamp):

        '''
        Generates a pose stamped message based on a vertex of a contour. Required to create a path message.
        '''

        # generates a pose stamped message by using the vertices of the safe set
        pose_stamp_message = PoseStamped()

        # set the header
        pose_stamp_message.header.frame_id = "odom" # this is the frame that the path will be visualized in Rviz
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

        '''
        Creates a path message composed of points from the safe set or obstacle contours' vertices
        '''

        # create a Path message
        msg = Path()

        # generate a time stamp for the path message
        time_stamp = self.get_clock().now().to_msg()

        # set the header
        msg.header.frame_id = "odom" # this is the frame that the path will be visualized in Rviz
        msg.header.stamp = time_stamp        

        # set the poses
        for i in range(len(points)):
            msg.poses.append(self.generate_pose_stamp(points[i], time_stamp))

        return msg

    def publish_safe_set(self, paths):

        '''
        Publishes the safe set with up to 5 holes as path messages for visualization in Rviz.
        '''

        # create an empty path message
        vertices = []
        msg = self.create_path_msg(vertices)

        # clear prior unused safe set publishers by publishing an empty message in their place, if there are any
        for i in range(5, len(paths)-1, -1):
            
            if i == 0:
                print("clearing safe set 1")
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

            vertices = swap_x_y_coordinates(vertices)

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
                self.get_logger().info('Safe set hole limit reached. Unable to visualize all safe set holes.')

            self.get_logger().info('Publishing: Safe Set Path')
    
    def publish_obstacles(self, paths):
     
        # clear the unused obstacle publishers if there are any

        # create an empty path message
        vertices = []
        msg = self.create_path_msg(vertices)

        # clear prior unused obstacle publishers by publishing an empty message in their place, if there are any
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

        self.get_logger().info('Publishing: Obstacle Paths')

        for i in range(len(paths)):
            
            path = paths[i]
            vertices = path.vertices

            vertices = swap_x_y_coordinates(vertices)

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

    def publish_initial_safe_set(self, vertices):

        vertices = swap_x_y_coordinates(vertices)

        # create a path message based on the initial safe set vertices
        msg = self.create_path_msg(vertices) 

        # publish the message
        self.get_logger().info('Publishing: Initial Safe Set Path')
        self.initial_safe_set_publisher_.publish(msg)

    def publish_goal_set(self, vertices):

        # create a path message based on the initial safe set vertices
        msg = self.create_path_msg(vertices) 

        # publish the message
        self.get_logger().info('Publishing: Goal Set Path')
        self.goal_set_publisher_.publish(msg)

    '''MAIN LOOP'''
    def timer_callback(self):
        
        # ********* Visualization *********

        # initiate a figure for matplotlib to plot the contours - size is irrelevant
        fig, self.ax = plt.subplots(1, 1, figsize=(1,1 ))

        ## generate the initial cbf safe set contour and goal set
        if self.iteration == 0:
            safe_set_contour = self.ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.cbf[..., 0], levels=[0], colors='k')
            self.initial_safe_set_vertices = safe_set_contour.collections[0].get_paths()[0].vertices # retrieve vertices of the contour for Rviz visualization
            self.publish_initial_safe_set(self.initial_safe_set_vertices)   # publish the vertices as a path message

            # generate the goal set vertices
            self.goal_set_vertices = generate_circle_vertices(radius = GOAL_SET_RADIUS, center = GOAL_SET_CENTER, num_vertices = GOAL_SET_VERTEX_COUNT)
            self.publish_goal_set(self.goal_set_vertices) # publish the goal set vertices as a path message
        
        # update the obstacle set based on the current refine CBF iteration
        update_obstacle_set(self, OBSTACLE_LIST, OBSTACLE_ITERATION_LIST)

        # publish obstacle contours as a path message
        self.obst_contour = self.ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.obstacle[..., 0], levels=[0], colors='r')
        obstacle_paths = self.obst_contour.collections[0].get_paths()
        self.publish_obstacles(obstacle_paths)

        # plot safe-set contours based on the nearest theta slice to the current heading angle
        theta_slice = self.grid.nearest_index(self.state)[2] 

        # TODO: fix this hacky solution
        if theta_slice == GRID_RESOLUTION[2]:
            # wrap theta slice to 0, as max grid point index does not exist
            theta_slice = 0

        print("Depicting safe set at theta slice: ", theta_slice)

        # publish safe set contours as a path message
        safe_set_contour = self.ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.cbf[..., theta_slice], levels=[0], colors='k')
        self.safe_set_vertices = safe_set_contour.collections[0].get_paths()[0].vertices # retrieve vertices of the contour for Rviz visualization
        safe_set_paths = safe_set_contour.collections[0].get_paths()
        self.publish_safe_set(safe_set_paths)

        # close the figure to prevent memory leak
        plt.close(fig)

def main():

    rclpy.init()
    refinecbf_visualization = RefineCBF_Visualization()

    try:
        refinecbf_visualization.get_logger().info("Starting visualization node, shut down with CTRL+C")
        rclpy.spin(refinecbf_visualization)
    
    except KeyboardInterrupt:
        refinecbf_visualization.get_logger().info('Keyboard interrupt, shutting down and saving CBF.\n')
        # save the cbf dictionary
        f = open('./log/cbf_to_save.pkl', 'wb')
        pickle.dump(refinecbf_visualization.cbf_to_save, f)
        f.close()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    refinecbf_visualization.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

