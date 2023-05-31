 # utils file for the refine_cbf experiment

# Experiment specific packages
import hj_reachability as hj
import refine_cbfs
from refine_cbfs.dynamics import HJControlAffineDynamics
from cbf_opt import ControlAffineDynamics, ControlAffineCBF
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import math

# ROS2 packages
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry

# Dynamics
# class for the dynamics of the differential drive robot
# inheritance: ControlAffineDynamics class from the cbf_opt package
class DiffDriveDynamics(ControlAffineDynamics):
    STATES   = ['X', 'Y', 'THETA'] # state vector of f(x,u,d): x
    CONTROLS = ['VEL', 'OMEGA']    # control vector of f(x,u,d): u
    DISTURBANCE = ['D']            # disturbance vector of f(x,u,d): d
    
    # parameterized constructor: dictionary object params, bool test, **kwargs
    # **kwargs allows a variable number of keyword arguments to be passed to a Python function
    def __init__(self, params, test = False, **kwargs):
        # assign a list with value 2 to the params key "periodic_dims"
        params["periodic_dims"] = [2]
        # pass the dict params, bool test to the base class constructor ControlAffineDynamics
        super().__init__(params, test, **kwargs)

    # class method: accepts a numpy array state and float time
    def open_loop_dynamics(self, state, time=0.0):
        return np.zeros_like(state) # f(x) returns an array of zeros with the same shape and type as a given array

    # class method: accepts numpy array state and float time
    def control_matrix(self, state, time=0.0):
        # numpy.repeat(a, repeats, axis = None)
        # repeats the matrix a, by the number of times specified, along the axis specified
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        # defining the B matrix based on dynamics
        B[..., 0, 0] = np.cos(state[..., 2])
        B[..., 1, 0] = np.sin(state[..., 2])
        B[..., 2, 1] = 1
        return B # returns the control matrix B from f(x,u,d) = f(x) + Bu + Cd

    # class method: accepts numpy array of state and float time
    def disturbance_jacobian(self, state, time=0.0):
        # C matrix
        C = np.repeat(np.zeros_like(state)[..., None], self.disturbance_dims, axis=-1)
        C[..., 0, 0] = 1
        return C # returns the disturbance matrix C from f(x,u,d) = f(x) + Bu + Cd
        # # returns an array of zeros equal to the dimension of the state
        # return np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)
    
    # class method: accepts numpy arrays state & control, and float time
    def state_jacobian(self, state, control, time=0.0):
        J = np.repeat(np.zeros_like(state)[..., None], self.n_dims, axis=-1) # axis = -1 means the last dimension
        J[..., 0, 2] = -control[..., 0] * np.sin(state[..., 2])
        J[..., 1, 2] = control[..., 0] * np.cos(state[..., 2])
        # returns a numpy array of the linearized dynamics based on the Jacobian of our system dynamics
        return J

# jax numpy version of the DiffDriveDynamics class
# inheritance: DiffDriveDyanmics from above
class DiffDriveJNPDynamics(DiffDriveDynamics):
    
    # class method: accepts numpy array of state and float time
    def open_loop_dynamics(self, state, time=0.0):
        # returns a jnp.array of 3 columns of zeros?
        return jnp.array([0., 0., 0.])
    
    # class method: accepts numpy array of state and float time
    def control_matrix(self, state, time=0.0):
        # returns a jax numpy version of the control matrix B*u
        return jnp.array([[jnp.cos(state[2]), 0],
                          [jnp.sin(state[2]), 0],
                          [0, 1]])

    # class method: accepts numpy array of state and float time
    def disturbance_jacobian(self, state, time=0.0):
        # returns a jax numpy array of 3x1 column of zeros
        return jnp.expand_dims(jnp.zeros(3), axis=-1)

    # class method: accepts numpy arrays state & control, and float time
    def state_jacobian(self, state, control, time=0.0):
        # returns a numpy array of the linearized dynamics based on the Jacobian of our system dynamics
        return jnp.array([
            [0, 0, -control[0] * jnp.sin(state[2])],
            [0, 0, control[0] * jnp.cos(state[2])], 
            [0, 0, 0]])

# Define a class called DiffDriveCBF
class DiffDriveCBF(ControlAffineCBF):
    # constructor: accepts and enforces types of parameters: dynamics, params
    # enforced to return None
    def __init__(self, dynamics: DiffDriveDynamics, params: dict = dict(), **kwargs) -> None:
        # define center, radius and scalar attributes from the dictionary argument
        self.center = params["center"]  # center of the circle defined by 0-superlevel set of h(x)
        self.r = params["r"]            # radius of the circle defined by 0-superlevel set of h(x)
        self.scalar = params["scalar"]  # scalar multipler of h(x)

        # call constructor from the super(base) class ControlAffineCBF
        super().__init__(dynamics, params, **kwargs)

    # h(x) (Can also be viewed as a value function as this is a CBVF)
    # class method: value function, accepts a numpy array of the state vector and float time
    # used to warmstart dynamic programming portion of HJR for the CBF
    def vf(self, state, time=0.0):
        # returns scalar*(r^2 - (x_state-x_center)^2 - (y_state - y_center)^2)
        return self.scalar * (self.r ** 2 - (state[..., 0] - self.center[0]) ** 2 - (state[..., 1] - self.center[1]) ** 2)

    # del_h(x) (Can also be viewed as the gradient of a value function since this is a CBVF)
    # class method: gradient of value function
    def _grad_vf(self, state, time=0.0):
        # make numpy array of same shape as state populated with 0's
        dvf_dx = np.zeros_like(state)
        # first row of gradient is equal to derivative w.r.t. x
        # -2 (x_state - x_circle)
        dvf_dx[..., 0] = -2 * (state[..., 0] - self.center[0])
        # second row of gradient is equal to the derivative w.r.t. y
        # -2 (y_state - y_circle)
        dvf_dx[..., 1] = -2 * (state[..., 1] - self.center[1])
        # returns the gradient times the constant that was factored out
        return self.scalar * dvf_dx

def euler_from_quaternion(x,y,z,w):

    "Converts a quaternion into euler angles (roll, pitch, yaw) in radians"

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def snap_state_to_grid_index(state, grid):

    "Snap a state to the nearest grid index and return it as a tuple."
    "Note: this functionality seems to already be implemented in the Grid class. Keeping here for now."

    grid_index = np.array(grid.nearest_index(state))

    # TODO: Fix this hacky workaround
    if grid_index[2] == grid.shape[2]:
        # floor the weird upper limit snap
        grid_index[2] = grid.shape[2]-1
    
    # return grid_index as a tuple
    return tuple(grid_index)
   
# data visualization
class ParameterStorage:
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.theta = np.array([])
        self.safety_value = np.array([])
        self.v = np.array([])
        self.omega = np.array([])
        self.v_nom = np.array([])
        self.omega_nom = np.array([])

    def append(self, x, y, theta, safety_value, v, omega, v_nom, omega_nom):
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.theta = np.append(self.theta, theta)
        self.safety_value = np.append(self.safety_value, safety_value)
        self.v = np.append(self.v, v)
        self.omega = np.append(self.omega, omega)
        self.v_nom = np.append(self.v_nom, v_nom)
        self.omega_nom = np.append(self.omega_nom, omega_nom)

    def plot_x(self):
        plt.plot(self.x)
        plt.title('X')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_y(self):
        plt.plot(self.y)
        plt.title('Y')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_theta(self):
        plt.plot(self.theta)
        plt.title('Theta')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_safety_value(self):
        plt.plot(self.safety_value)
        plt.title('Safety Value')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_v(self):
        plt.plot(self.v)
        plt.title('V')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_omega(self):
        plt.plot(self.omega)
        plt.title('Omega')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_v_nom(self):
        plt.plot(self.v_nom)
        plt.title('V Nom')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_omega_nom(self):
        plt.plot(self.omega_nom)
        plt.title('Omega Nom')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show() 

    def plot_all(self):
        fig, axs = plt.subplots(2, 4, figsize=(14, 8))
        axs[0, 0].plot(self.x)
        axs[0, 0].set_title('X')
        axs[0, 1].plot(self.y)
        axs[0, 1].set_title('Y')
        axs[0, 2].plot(self.theta)
        axs[0, 2].set_title('Theta')
        axs[1, 0].plot(self.safety_value)
        axs[1, 0].set_title('Safety Value')
        axs[1, 1].plot(self.v)
        axs[1, 1].set_title('V')
        axs[1, 2].plot(self.omega)
        axs[1, 2].set_title('Omega')
        axs[0, 3].plot(self.v_nom)
        axs[0, 3].set_title('V Nominal')
        axs[1, 3].plot(self.omega_nom)
        axs[1, 3].set_title('Omega Nominal')
        plt.tight_layout()
        plt.show()

    # save data to a csv file
    def save_data(self, filename):
        np.savetxt(filename, np.c_[self.x, self.y, self.theta, self.safety_value, self.v, self.omega, self.v_nom, self.omega_nom], delimiter=',')
        print("Data saved to " + filename)

    # load data from a csv file
    def load_data(self, filename):
        data = np.genfromtxt(filename, delimiter=',')
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.theta = data[:, 2]
        self.safety_value = data[:, 3]
        self.v = data[:, 4]
        self.omega = data[:, 5]
        self.v_nom = data[:, 6]
        self.omega_nom = data[:, 7]
        print("Data loaded from " + filename)

# OBSTACLE TYPE

# Options [x] means current functionality implemented:
#  [x] Rectangular
#  [x] Circular
#  [x] Elliptical
#  [x] Bounding Box
#  [ ] Line
#  [ ] Point
#  [ ] Rectangular Prism
#  [ ] Cylinder
#  [ ] Sphere (could remove circular?)
#  [ ] Ellipsoid (could remove elliptical?)
#  [ ] Union
#  [ ] Polygonal
#  [ ] ...

def circle_constraint(state, center=np.array([1,1]), radius=1, padding=0):

    distance = jnp.linalg.norm(state[:2] - center) - padding

    return distance - radius

def rectangle_constraint(state, center=np.array([1,1]), length=1, padding=0):
    
    # TODO: This needs to not only return a square, but also a rectangle

    # extend the length of the square by the padding
    length_with_padding = length + 2 * padding

    # coordinate of bottom left corner of the obstacle
    bottom_left = jnp.array(center - length_with_padding / 2)
    
    # returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
    return -jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length_with_padding[0] - state[0], 
                            state[1] - bottom_left[1], bottom_left[1] + length_with_padding[1] - state[1]]))

def bounding_box_constraint(state, center, length=2, padding=0):
    
    # extend the length of the square by the padding
    length_with_padding = length - 2 * padding

    # coordinate of bottom left corner of the obstacle
    bottom_left = jnp.array(center - length_with_padding / 2)

    # returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
    return jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length_with_padding[0] - state[0], 
                            state[1] - bottom_left[1], bottom_left[1] + length_with_padding[1] - state[1]]))

def create_circle_constraint(center, radius, padding):
    
        def circle_constraint(state):
    
            distance = jnp.linalg.norm(state[:2] - center) - padding
    
            return distance - radius
    
        return circle_constraint

# def create_rectangle_constraint(center, length, padding):
    
#         def rectangle_constraint(state):

#             # extend the length of the square by the padding
#             length_with_padding = length + 2 * padding

#             # coordinate of bottom left corner of the obstacle
#             bottom_left = jnp.array(center - length_with_padding / 2)
            
#             # returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
#             return -jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length_with_padding[0] - state[0], 
#                                     state[1] - bottom_left[1], bottom_left[1] + length_with_padding[1] - state[1]]))
        
#         return rectangle_constraint

# def create_bounding_box_constraint(center, length, padding):
    
#         def bounding_box_constraint(state):

#             # retract the length of the square by the padding
#             length_with_padding = length - 2 * padding

#             # coordinate of bottom left corner of the obstacle
#             bottom_left = jnp.array(center - length_with_padding / 2)

#             # returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
#             return jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length_with_padding[0] - state[0], 
#                                     state[1] - bottom_left[1], bottom_left[1] + length_with_padding[1] - state[1]]))
        
#         return bounding_box_constraint


def create_rectangle_constraint(center, length, padding):
    """
    Creates a rectangle constraint function based on the given parameters.

    Parameters:
    center: The center coordinates of the rectangle.
    length: The length of the rectangle.
    padding: The padding value for the rectangle.

    Returns:
    A rectangle constraint function.
    """
    def rectangle_constraint(state):
        # Extend the length of the square by the padding
        length_with_padding = length + 2 * padding

        # Coordinate of bottom left corner of the obstacle
        bottom_left = jnp.array(center - length_with_padding / 2)

        # Returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
        return -jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length_with_padding[0] - state[0],
                                   state[1] - bottom_left[1], bottom_left[1] + length_with_padding[1] - state[1]]))

    return rectangle_constraint


def create_bounding_box_constraint(center, length, padding):
    """
    Creates a bounding box constraint function based on the given parameters.

    Parameters:
    center: The center coordinates of the bounding box.
    length: The length of the bounding box.
    padding: The padding value for the bounding box.

    Returns:
    A bounding box constraint function.
    """
    def bounding_box_constraint(state):
        # Retract the length of the square by the padding
        length_with_padding = length - 2 * padding

        # Coordinate of bottom left corner of the obstacle
        bottom_left = jnp.array(center - length_with_padding / 2)

        # Returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
        return jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length_with_padding[0] - state[0],
                                  state[1] - bottom_left[1], bottom_left[1] + length_with_padding[1] - state[1]]))

    return bounding_box_constraint


def define_constraint_set(obstacles, padding):

    """
    Defines a constraint set l(x) based on obstacles provided.

    Args:
        obstacles : A dictionary of obstacles with the key being the obstacle type and the value being a dictionary of obstacle parameters
        padding : Float that inflates the obstacles by a certain amount using Minkoswki sum

    Returns:
        A function that is a constraint set l(x) based on the given obstacles for use in Python HJ Reachability package. Takes current state as argument and
        returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise).

    Note: Only works with differential drive dynamics currently.
    """

    constraints_list = [] # list of constraints

    # loop through obstacles dictionary and define constraint set based on obstacle type
    for obstacle_type in obstacles:
            
        # if obstacle type is circular
        if obstacle_type == "circle":

            # loop through circle dictionary and define constraint set
            for circle in obstacles[obstacle_type]:

                constraint = create_circle_constraint(obstacles[obstacle_type][circle]["center"], obstacles[obstacle_type][circle]["radius"], padding)
                constraints_list.append(constraint)

        elif obstacle_type == "bounding_box":

            for bounding_box in obstacles[obstacle_type]:

                constraint = create_bounding_box_constraint(obstacles[obstacle_type][bounding_box]["center"], obstacles[obstacle_type][bounding_box]["length"], padding)
                constraints_list.append(constraint)

        elif obstacle_type == "rectangle":

            for rectangle in obstacles[obstacle_type]:

                constraint = create_rectangle_constraint(obstacles[obstacle_type][rectangle]["center"], obstacles[obstacle_type][rectangle]["length"], padding)
                constraints_list.append(constraint)

        else: # if obstacle type is not supported yet
            raise NotImplementedError("Obstacle type is not supported yet.")
        
        # append to list of constraints
        #constraints_list.append(constraint)


    def constraint_set(state):

        """
        A real-valued function s.t. the zero-superlevel set is the safe set

        Args:
            state : An unbatched (!) state vector, an array of shape `(3,)` containing `[x, y, omega]`.

        Returns:
            A scalar, positive iff the state is in the safe set, negative or 0 otherwise.
        """

        # initialize numpy array of constraints
        numpy_array_of_constraints = np.array([])

        # loop through list of constraints
        for l in constraints_list:
                
            # append constraint to numpy array
            numpy_array_of_constraints = jnp.append(numpy_array_of_constraints, l(state))

        # loop through numpy array of constraints and take the piecewise minimum
        return jnp.min(numpy_array_of_constraints)
        
    return constraint_set


def save_float_to_file(data, filename):
    first_call = False
    
    try:
        with open(filename, 'r') as file:
            first_call = file.read() == ''
    except FileNotFoundError:
        first_call = True
    
    mode = 'w' if first_call else 'a'
    
    with open(filename, mode) as file:
        file.write(str(data) + '\n')


def state_feedback_config_error():

    '''
    Prints an error message to the console if the state feedback configuration is not properly configured.
    For example, if one of the GLOBAL state config variables is set to something other than True or False.
    '''

    print("Error: State feedback not properly configured. Please check config.py file.")
    exit()

def configure_state_feedback_subscriber(self, qos, topic_string: str):

    '''
    Assigns the state feedback subscriber based on the configuration in config.py
    to the self.state_sub attribute of the ROS node class instance.
    '''
    print("Logic check:", topic_string == 'vicon/turtlebot_1/turtlebot_1')

    if topic_string == 'gazebo/odom' or topic_string == 'odom':
        state_sub = self.create_subscription(
            Odometry,
            topic_string,
            self.state_sub_callback,
            qos)
        
    elif topic_string == 'vicon/turtlebot_1/turtlebot_1':
        state_sub = self.create_subscription(
            TransformStamped,
            topic_string,
            self.state_sub_callback_vicon,
            qos)
        
    else:
        state_feedback_config_error()        

    return state_sub


def configure_nominal_policy_publisher(self, qos, USE_UNFILTERED_POLICY):
    
    if USE_UNFILTERED_POLICY is True:
        nom_pol_publisher_ = self.create_publisher(
                    Twist, 
                    'cmd_vel',
                    qos)
    
    elif USE_UNFILTERED_POLICY is False:
        nom_pol_publisher_ = self.create_publisher(
                    Twist, 
                    'nom_policy',
                    qos)
        
    else:
        print("Error: Nominal policy publisher not properly configured. Please check config.py file.")
        exit()

    return nom_pol_publisher_


def compute_nominal_control(self):

    nominal_policy = self.grid.interpolate(self.nominal_policy_table, self.state)
    nominal_policy = np.reshape(nominal_policy, (1, self.dyn.control_dims))

    return nominal_policy

def create_nominal_policy_publishing_message(USE_UNFILTERED_POLICY):

    if USE_UNFILTERED_POLICY is True:
        nominal_policy_message = 'Publishing nominal control input over topic /cmd_vel.'
    
    elif USE_UNFILTERED_POLICY is False:
        nominal_policy_message = 'Publishing nominal control input over topic /nom_policy.'  
    else:
        print("Error: USE_UNFILTERED_POLICY is not configured correctly. Please check config.py file.")

    return nominal_policy_message


def create_new_obstacle_set(self, obstacles, iteration):

    print('New obstacle introduced at iteration: ', iteration)

    # redifine the constraint set
    self.constraint_set = define_constraint_set(obstacles, self.obst_padding)
    

def introduce_obstacle(self, OBSTACLE_LIST, OBSTACLE_ITERATION_LIST):

    # introduce a new constraint set at specified iteration
    if self.iteration == OBSTACLE_ITERATION_LIST[0]:        
        create_new_obstacle_set(self, OBSTACLE_LIST[1], OBSTACLE_ITERATION_LIST[0])

    elif self.iteration == OBSTACLE_ITERATION_LIST[1]:        
        create_new_obstacle_set(self, OBSTACLE_LIST[2], OBSTACLE_ITERATION_LIST[1])

    elif self.iteration == OBSTACLE_ITERATION_LIST[2]:
        create_new_obstacle_set(self, OBSTACLE_LIST[3], OBSTACLE_ITERATION_LIST[2])

    elif self.iteration == OBSTACLE_ITERATION_LIST[3]:
        create_new_obstacle_set(self, OBSTACLE_LIST[4], OBSTACLE_ITERATION_LIST[3])

    else:
        return

    # redefine the obstacle
    self.obstacle = hj.utils.multivmap(self.constraint_set, jnp.arange(self.grid.ndim))(self.grid.states)
    # redefine the brt function
    brt_fct = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))  # Backwards reachable TUBE!
    # redefine the solver settings
    self.solver_settings = hj.SolverSettings.with_accuracy("high", value_postprocessor=brt_fct(self.obstacle))


def update_obstacle_set(self, OBSTACLE_LIST, OBSTACLE_ITERATION_LIST):

    # introduce a new constraint set at specified iteration
    if self.iteration == OBSTACLE_ITERATION_LIST[0]:        
        create_new_obstacle_set(self, OBSTACLE_LIST[1], OBSTACLE_ITERATION_LIST[0])

    elif self.iteration == OBSTACLE_ITERATION_LIST[1]:        
        create_new_obstacle_set(self, OBSTACLE_LIST[2], OBSTACLE_ITERATION_LIST[1])

    elif self.iteration == OBSTACLE_ITERATION_LIST[2]:
        create_new_obstacle_set(self, OBSTACLE_LIST[3], OBSTACLE_ITERATION_LIST[2])

    elif self.iteration == OBSTACLE_ITERATION_LIST[3]:
        create_new_obstacle_set(self, OBSTACLE_LIST[4], OBSTACLE_ITERATION_LIST[3])

    # redefine the obstacle
    self.obstacle = hj.utils.multivmap(self.constraint_set, jnp.arange(self.grid.ndim))(self.grid.states)

def swap_x_y_coordinates(vertices):

    # swap the x and y coordinates to reflect the grid rotation from Python to Rviz / Gazebo
    for j in range(len(vertices)):
        temp = vertices[j][0]
        vertices[j][0] = vertices[j][1]
        vertices[j][1] = temp

    return vertices

def generate_circle_vertices(radius, num_vertices, center=(0, 0)):
    vertices = np.zeros((num_vertices + 1, 2))  # Increased size by 1
    angle_increment = 2 * math.pi / num_vertices

    for i in range(num_vertices):
        angle = i * angle_increment
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices[i] = [x, y]

    # Duplicate initial vertex
    vertices[-1] = vertices[0]

    return vertices