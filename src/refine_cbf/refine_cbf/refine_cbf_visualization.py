#!/usr/bin/env python3

# Node which sends visualization topics to rviz

# NOTE: There will be a slight delay in the visuals versus what the robot actually "sees". This is why it may appear the system 
#       leaves the current safe-set in RVIZ.
# TODO: Occasionally, this script crashes due to the loading of the new cbf happening as it is being saved to. Should try and find a workaround.

from rclpy.node import Node
from refine_cbf.config import *

class RefineCBFVisualization(Node):
    """
    RVIZ Visualization Node for refineCBF.

    Subscribed topics: \gazebo\odom (or \odom or \vicon_odom), \cbf_availability
    Published topics: \obstacle_1, \obstacle_2, \obstacle_3, \obstacle_4, \obstacle_5, \safe_set_1,
                        \safe_set_2, \safe_set_3, \safe_set_4, \safe_set_5, \initial_safe_set, \goal_set
    """
        
    def __init__(self):
        """
        Initializes the RefineCBFVisualization node.
        """
        # Node name in ROS2 network
        super().__init__('visualizer')

        # See config.py file for GLOBAL variable definitions
        self.grid = GRID
        self.obst_padding = OBSTACLE_PADDING # used when creating new constraint function
        self.constraint_set = define_constraint_function(OBSTACLE_LIST[0], OBSTACLE_PADDING)  # Defining the constraint set l(x) terminal cost
        self.obstacle = hj.utils.multivmap(self.constraint_set, jnp.arange(self.grid.ndim))(self.grid.states)  # Defining the obstacle for the Hamilton Jacobi Reachability package

        # A depth of 10 suffices in most cases, but this can be increased if needed
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

        timer_period = 0.005  # Delay between starting iterations [seconds]

        self.timer = self.create_timer(timer_period, self.timer_callback)  # Calls the timer_callback function every timer_period
        self.cbf_available = False  # Set cbf availability to false initially
        
        self.safe_set_vertices = []
        self.initial_safe_set_vertices = []

        self.iteration = 0

        if USE_REFINECBF:
            self.cbf = INITIAL_CBF
        else:
            self.cbf = np.load(PRECOMPUTED_CBF_FILENAME)

        self.initial_cbf = self.cbf  # Save the initial cbf for visualization purposes
            
        # Cbf to save, a dictionary where new entries are made upon receiving a new CBF
        self.cbf_to_save = {self.iteration: self.cbf}
        
        # Create the state feedback subscriber callback function
        create_sub_callback(self.__class__, STATE_FEEDBACK_TOPIC)

        # Initial state
        self.state = INITIAL_STATE

        self.state_sub = configure_state_feedback_subscriber(self, qos, topic_string=STATE_FEEDBACK_TOPIC)

        # Cbf flag subscriber
        self.cbf_avail_sub = self.create_subscription(
            Bool,
            'cbf_availability',
            self.cbf_sub_callback,
            qos)

    # Boolean flag availability subscription
    # NOTE: If this proves to be too slow for denser grids and higher-dimensional state spaces,
    # replace this with float64 message of the matrix? 
    def cbf_sub_callback(self, bool_msg):

        self.get_logger().info('New CBF Available: "%s"' % bool_msg.data)

        if bool_msg.data:
            
            # Increment the counter
            self.iteration += 1

            # Halt the robot to prevent unsafe behavior while the CBF is updated
            self.get_logger().info('New CBF received, loading new CBF')

            # Load new tabular cbf
            self.cbf = jnp.load('./log/cbf.npy')

            # Update the cbf to save by adding new iteration to the dictionary
            self.cbf_to_save[self.iteration] = self.cbf

    def generate_pose_stamp(self, vertex, time_stamp):
        """
        Generates a pose-stamped message based on a vertex of a contour. Required to create a path message.

        Args:
            vertex: The vertex of the contour.
            time_stamp: The time stamp for the path message.

        Returns:
            The generated PoseStamped message.
        """
        # Generates a pose-stamped message using the vertices of the safe set
        pose_stamp_message = PoseStamped()

        # Set the header
        pose_stamp_message.header.frame_id = "odom"  # This is the frame that the path will be visualized in Rviz
        pose_stamp_message.header.stamp = time_stamp
        
        # Set the pose
        
        # Set the position
        pose_stamp_message.pose.position.x = vertex[0]
        pose_stamp_message.pose.position.y = vertex[1]
        pose_stamp_message.pose.position.z = 0.0  # Project onto the ground

        # Set the orientation (this will be irrelevant, as we only care about the position)
        pose_stamp_message.pose.orientation.x = 0.0
        pose_stamp_message.pose.orientation.y = 0.0
        pose_stamp_message.pose.orientation.z = 0.0
        pose_stamp_message.pose.orientation.w = 1.0

        return pose_stamp_message
    
    def create_path_msg(self, points):
        """
        Creates a path message composed of points from the safe set or obstacle contours' vertices.

        Args:
            points: The points representing the path.

        Returns:
            The generated Path message.
        """
        # Create a Path message
        msg = Path()

        # Generate a time stamp for the path message
        time_stamp = self.get_clock().now().to_msg()

        # Set the header
        msg.header.frame_id = "odom"  # This is the frame that the path will be visualized in Rviz
        msg.header.stamp = time_stamp        

        # Set the poses
        for i in range(len(points)):
            msg.poses.append(self.generate_pose_stamp(points[i], time_stamp))

        return msg

    def publish_safe_set(self, paths):
        """
        Publishes the safe set with up to 5 holes as path messages for visualization in Rviz.

        Args:
            paths: The paths representing the safe set.
        """
        # Create an empty path message
        vertices = []
        msg = self.create_path_msg(vertices)

        # Clear prior unused safe set publishers by publishing an empty message in their place, if there are any
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
        """
        Publishes the obstacles as path messages for visualization in Rviz.

        Args:
            paths: The paths representing the obstacles.
        """
        # Clear the unused obstacle publishers if there are any

        # Create an empty path message
        vertices = []
        msg = self.create_path_msg(vertices)

        # Clear prior unused obstacle publishers by publishing an empty message in their place, if there are any
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

        # Publish the obstacle paths
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
        """
        Publishes the initial safe set vertices as a path message for visualization in Rviz.

        Args:
            vertices: The vertices representing the initial safe set.
        """
        vertices = swap_x_y_coordinates(vertices)

        # Create a path message based on the initial safe set vertices
        msg = self.create_path_msg(vertices) 

        # Publish the message
        self.get_logger().info('Publishing: Initial Safe Set Path')
        self.initial_safe_set_publisher_.publish(msg)

    def publish_goal_set(self, vertices):
        """
        Publishes the goal set vertices as a path message for visualization in Rviz.

        Args:
            vertices: The vertices representing the goal set.
        """
        # Create a path message based on the goal set vertices
        msg = self.create_path_msg(vertices) 

        # Publish the message
        self.get_logger().info('Publishing: Goal Set Path')
        self.goal_set_publisher_.publish(msg)

    '''MAIN LOOP'''
    def timer_callback(self):
        """
        The main loop callback function that performs visualization updates.
        """
        # ********* Visualization *********

        # Initiate a figure for matplotlib to plot the contours - size is irrelevant
        fig, self.ax = plt.subplots(1, 1, figsize=(1, 1))

        ## Generate the goal set just once
        if self.iteration == 0:
            
            # Generate the goal set vertices
            self.goal_set_vertices = generate_circle_vertices(radius=GOAL_SET_RADIUS, center=GOAL_SET_CENTER, num_vertices=50)
            self.publish_goal_set(self.goal_set_vertices)  # Publish the goal set vertices as a path message
        
        # Update the obstacle set based on the current refine CBF iteration
        update_obstacle_set(self, OBSTACLE_LIST, OBSTACLE_ITERATION_LIST)

        # Publish obstacle contours as a path message
        self.obst_contour = self.ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.obstacle[..., 0], levels=[0], colors='r')
        obstacle_paths = self.obst_contour.collections[0].get_paths()
        self.publish_obstacles(obstacle_paths)

        # Plot safe-set contours based on the nearest theta slice to the current heading angle
        theta_slice = self.grid.nearest_index(self.state)[2] 

        # TODO: fix this hacky solution
        if theta_slice == GRID_RESOLUTION[2]:
            # Wrap theta slice to 0, as the max grid point index does not exist
            theta_slice = 0

        print("Depicting safe set at theta slice: ", theta_slice)

        # Initial safe set contour
        initial_safe_set_contour = self.ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.initial_cbf[..., theta_slice], levels=[0], colors='k')
        self.initial_safe_set_vertices = initial_safe_set_contour.collections[0].get_paths()[0].vertices  # Retrieve vertices of the contour for Rviz visualization
        self.publish_initial_safe_set(self.initial_safe_set_vertices)  # Publish the vertices as a path message

        # Publish safe set contours as a path message
        safe_set_contour = self.ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.cbf[..., theta_slice], levels=[0], colors='k')
        self.safe_set_vertices = safe_set_contour.collections[0].get_paths()[0].vertices  # Retrieve vertices of the contour for Rviz visualization
        safe_set_paths = safe_set_contour.collections[0].get_paths()
        self.publish_safe_set(safe_set_paths)

        # Close the figure to prevent memory leak
        plt.close(fig)

def main():
    """
    Main function to initialize the RefineCBFVisualization node.
    """
    rclpy.init()
    refinecbf_visualization = RefineCBFVisualization()

    try:
        refinecbf_visualization.get_logger().info("Starting visualization node, shut down with CTRL+C")
        rclpy.spin(refinecbf_visualization)
    
    except KeyboardInterrupt:
        refinecbf_visualization.get_logger().info('Keyboard interrupt, shutting down and saving CBF.\n')
        # Save the cbf dictionary
        f = open('./experiment_cbf.pkl', 'wb')
        pickle.dump(refinecbf_visualization.cbf_to_save, f)
        f.close()

    # Destroy the node explicitly (optional - otherwise it will be done automatically when the garbage collector destroys the node object)
    refinecbf_visualization.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
