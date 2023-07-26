# Utils file for the refine_cbf package

# Experiment-specific packages
import hj_reachability as hj
from cbf_opt import ControlAffineDynamics, ControlAffineCBF
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import math
from typing import Type

# ROS2 packages
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry


class DiffDriveDynamics(ControlAffineDynamics):
    """Class for the dynamics of the differential drive robot.
    
    Inherits from the ControlAffineDynamics class in the cbf_opt package.

    Attributes:
        STATES (list): State vector of f(x,u,d): x.
        CONTROLS (list): Control vector of f(x,u,d): u.
        DISTURBANCE (list): Disturbance vector of f(x,u,d): d.
    """

    STATES   = ['X', 'Y', 'THETA']
    CONTROLS = ['VEL', 'OMEGA']
    DISTURBANCE = ['D']

    def __init__(self, params: dict, test=False, **kwargs):
        """Parameterized constructor.

        Args:
            params (dict): Dictionary object containing parameters.
            test (bool): Boolean flag indicating test mode.
            **kwargs: Variable number of keyword arguments.
        """
        params["periodic_dims"] = [2]
        super().__init__(params, test, **kwargs)

    def open_loop_dynamics(self, state, time=0.0):
        """Open-loop dynamics method.

        Args:
            state (numpy.array): Array representing the state.
            time (float, optional): Time value. Defaults to 0.0.

        Returns:
            numpy.array: Array of zeros with the same shape as the input state.
        """
        return np.zeros_like(state)

    def control_matrix(self, state, time=0.0):
        """Control matrix method.

        Args:
            state (numpy.array): Array representing the state.
            time (float, optional): Time value. Defaults to 0.0.

        Returns:
            numpy.array: Control matrix B from f(x,u,d) = f(x) + Bu + Cd.
        """
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 0, 0] = np.cos(state[..., 2])
        B[..., 1, 0] = np.sin(state[..., 2])
        B[..., 2, 1] = 1
        return B

    def disturbance_jacobian(self, state, time=0.0):
        """Disturbance jacobian method.

        Args:
            state (numpy.array): Array representing the state.
            time (float, optional): Time value. Defaults to 0.0.

        Returns:
            numpy.array: Disturbance matrix C from f(x,u,d) = f(x) + Bu + Cd.
        """
        C = np.repeat(np.zeros_like(state)[..., None], self.disturbance_dims, axis=-1)
        C[..., 0, 0] = 1
        return C

    def state_jacobian(self, state, control, time=0.0):
        """State jacobian method.

        Args:
            state (numpy.array): Array representing the state.
            control (numpy.array): Array representing the control.
            time (float, optional): Time value. Defaults to 0.0.

        Returns:
            numpy.array: Linearized dynamics based on the Jacobian of the system dynamics.
        """
        J = np.repeat(np.zeros_like(state)[..., None], self.n_dims, axis=-1)
        J[..., 0, 2] = -control[..., 0] * np.sin(state[..., 2])
        J[..., 1, 2] = control[..., 0] * np.cos(state[..., 2])
        return J


# JAX NumPy version of the DiffDriveDynamics class.
# Inherits from the DiffDriveDynamics class defined above.
class DiffDriveJNPDynamics(DiffDriveDynamics):
    """
    JAX NumPy version of the DiffDriveDynamics class.

    Inherits from the DiffDriveDynamics class defined above.
    """

    def open_loop_dynamics(self, state, time=0.0):
        """
        Open-loop dynamics method.

        Args:
            state (numpy.array): Array representing the state.
            time (float, optional): Time value. Defaults to 0.0.

        Returns:
            jnp.array: JAX NumPy array with three columns of zeros.
        """
        return jnp.array([0., 0., 0.])

    def control_matrix(self, state, time=0.0):
        """
        Control matrix method.

        Args:
            state (numpy.array): Array representing the state.
            time (float, optional): Time value. Defaults to 0.0.

        Returns:
            jnp.array: JAX NumPy version of the control matrix B*u.
        """
        return jnp.array([[jnp.cos(state[2]), 0],
                          [jnp.sin(state[2]), 0],
                          [0, 1]])

    def disturbance_jacobian(self, state, time=0.0):
        """
        Disturbance jacobian method.

        Args:
            state (numpy.array): Array representing the state.
            time (float, optional): Time value. Defaults to 0.0.

        Returns:
            jnp.array: JAX NumPy array with a 3x1 column of zeros.
        """
        return jnp.expand_dims(jnp.zeros(3), axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        """
        State jacobian method.

        Args:
            state (numpy.array): Array representing the state.
            control (numpy.array): Array representing the control.
            time (float, optional): Time value. Defaults to 0.0.

        Returns:
            jnp.array: JAX NumPy array representing the linearized dynamics based on the Jacobian of the system dynamics.
        """
        return jnp.array([
            [0, 0, -control[0] * jnp.sin(state[2])],
            [0, 0, control[0] * jnp.cos(state[2])],
            [0, 0, 0]
        ])


# Define a class called DiffDriveCBF.
class DiffDriveCBF(ControlAffineCBF):
    """
    Class representing the control barrier function for the differential drive robot.

    Inherits from the ControlAffineCBF class.
    """

    def __init__(self, dynamics: DiffDriveDynamics, params: dict = dict(), **kwargs) -> None:
        """
        Constructor method.

        Args:
            dynamics (DiffDriveDynamics): Dynamics of the differential drive robot.
            params (dict, optional): Dictionary containing parameters. Defaults to an empty dictionary.
            **kwargs: Variable number of keyword arguments.
        """
        self.center = params["center"]  # Center of the circle defined by 0-superlevel set of h(x)
        self.r = params["r"]            # Radius of the circle defined by 0-superlevel set of h(x)
        self.scalar = params["scalar"]  # Scalar multiplier of h(x)

        super().__init__(dynamics, params, **kwargs)

    def vf(self, state, time=0.0):
        """
        Value function (h(x)) method.

        Args:
            state (numpy.array): Array representing the state.
            time (float, optional): Time value. Defaults to 0.0.

        Returns:
            jnp.array: JAX NumPy array representing the value function.
        """
        return self.scalar * (self.r ** 2 - (state[..., 0] - self.center[0]) ** 2 - (state[..., 1] - self.center[1]) ** 2)

    def _grad_vf(self, state, time=0.0):
        """
        Gradient of the value function (del_h(x)) method.

        Args:
            state (numpy.array): Array representing the state.
            time (float, optional): Time value. Defaults to 0.0.

        Returns:
            jnp.array: JAX NumPy array representing the gradient of the value function.
        """
        dvf_dx = np.zeros_like(state)
        dvf_dx[..., 0] = -2 * (state[..., 0] - self.center[0])
        dvf_dx[..., 1] = -2 * (state[..., 1] - self.center[1])
        return self.scalar * dvf_dx


def euler_from_quaternion(x: float, y: float, z: float, w: float):
    """
    Converts a quaternion into Euler angles (roll, pitch, yaw) in radians.

    Arguments:
    x, y, z, w (float): Components of the quaternion.

    Returns:
    tuple: Euler angles (roll, pitch, yaw) in radians.
    """

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

    return roll_x, pitch_y, yaw_z  # in radians


def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    """
    Converts Euler angles to quaternion representation.

    Arguments:
    roll (float): The roll angle in radians.
    pitch (float): The pitch angle in radians.
    yaw (float): The yaw angle in radians.

    Returns:
    tuple: A tuple representing the quaternion in the order (w, x, y, z).
    """

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz


def shift_quaternion_by_yaw(quaternion: tuple, yaw_shift: float):
    """
    Shifts a quaternion angle representation by a yaw rotation.

    Arguments:
    quaternion (tuple): A tuple representing the original quaternion in the order (w, x, y, z).
    yaw_shift (float): The yaw rotation angle in radians.

    Returns:
    tuple: A tuple representing the shifted quaternion in the order (w, x, y, z).
    """

    # Convert the yaw_shift to a quaternion
    shift_quaternion = (
        math.cos(yaw_shift / 2),  # w
        0,  # x
        0,  # y
        math.sin(yaw_shift / 2)  # z
    )

    # Multiply the original quaternion by the shift quaternion
    shifted_quaternion = quaternion_multiply(shift_quaternion, quaternion)

    return shifted_quaternion


def quaternion_multiply(quat1: tuple, quat2: tuple):
    """
    Multiplies two quaternions.

    Arguments:
    quat1 (tuple): A tuple representing the first quaternion in the order (w, x, y, z).
    quat2 (tuple): A tuple representing the second quaternion in the order (w, x, y, z).

    Returns:
    tuple: A tuple representing the result of quaternion multiplication in the order (w, x, y, z).
    """

    w1, x1, y1, z1 = quat1
    w2, x2, y2, z2 = quat2

    result = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2   # z
    )

    return result


class ParameterStorage:
    """
    A class for storing and visualizing parameter data.

    Attributes:
        x (numpy array): Array to store 'X' parameter data.
        y (numpy array): Array to store 'Y' parameter data.
        theta (numpy array): Array to store 'Theta' parameter data.
        safety_value (numpy array): Array to store 'Safety Value' parameter data.
        v (numpy array): Array to store 'V' parameter data.
        omega (numpy array): Array to store 'Omega' parameter data.
        v_nom (numpy array): Array to store 'V Nominal' parameter data.
        omega_nom (numpy array): Array to store 'Omega Nominal' parameter data.
    """

    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.theta = np.array([])
        self.safety_value = np.array([])
        self.v = np.array([])
        self.omega = np.array([])
        self.v_nom = np.array([])
        self.omega_nom = np.array([])

    def append(self, x=0., y=0., theta=0., safety_value=0., v=0., omega=0., v_nom=0., omega_nom=0.):
        """
        Append data to respective parameter arrays.

        Arguments:
            x, y, theta, safety_value, v, omega, v_nom, omega_nom (float): Data to be appended to the arrays.
        """
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.theta = np.append(self.theta, theta)
        self.safety_value = np.append(self.safety_value, safety_value)
        self.v = np.append(self.v, v)
        self.omega = np.append(self.omega, omega)
        self.v_nom = np.append(self.v_nom, v_nom)
        self.omega_nom = np.append(self.omega_nom, omega_nom)

    def plot_x(self):
        """
        Plot the 'X' parameter data.
        """
        plt.plot(self.x)
        plt.title('X')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_y(self):
        """
        Plot the 'Y' parameter data.
        """
        plt.plot(self.y)
        plt.title('Y')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_theta(self):
        """
        Plot the 'Theta' parameter data.
        """
        plt.plot(self.theta)
        plt.title('Theta')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_safety_value(self):
        """
        Plot the 'Safety Value' parameter data.
        """
        plt.plot(self.safety_value)
        plt.title('Safety Value')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_v(self):
        """
        Plot the 'V' parameter data.
        """
        plt.plot(self.v)
        plt.title('V')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_omega(self):
        """
        Plot the 'Omega' parameter data.
        """
        plt.plot(self.omega)
        plt.title('Omega')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_v_nom(self):
        """
        Plot the 'V Nom' parameter data.
        """
        plt.plot(self.v_nom)
        plt.title('V Nominal')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_omega_nom(self):
        """
        Plot the 'Omega Nom' parameter data.
        """
        plt.plot(self.omega_nom)
        plt.title('Omega Nominal')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.show()

    def plot_nominal(self):
        """
        Plot 'V Nominal' and 'Omega Nominal' parameter data side by side.
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        axs[0].plot(self.v_nom)
        axs[0].set_title('V Nominal')
        axs[1].plot(self.omega_nom)
        axs[1].set_title('Omega Nominal')
        plt.tight_layout()
        plt.show()

    def plot_all(self):
        """
        Plot all parameter data in a 2x4 grid.
        """
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

    def save_data(self, filename: str):
        """
        Save parameter data to a CSV file.

        Arguments:
            filename (str): Name of the CSV file to save the data.
        """
        np.savetxt(filename, np.c_[self.x, self.y, self.theta, self.safety_value, self.v, self.omega, self.v_nom, self.omega_nom], delimiter=',')
        print("Data saved to " + filename)

    def load_data(self, filename: str):
        """
        Load parameter data from a CSV file.

        Arguments:
            filename (str): Name of the CSV file to load the data from.
        """
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


def create_circle_constraint(center: np.ndarray, radius: float, padding: float):
    """
    Creates a circular constraint function based on the given parameters.

    Parameters:
    center: 2x1 numpy array, the center coordinates of the circle
    radius: float, the radius of the circle.
    padding: float, the padding value around the circle.

    Returns:
    A circle constraint function.
    """
    def circle_constraint(state: np.ndarray):
        """
        Circular constraint function that computes the distance from the state to the circular constraint boundary.

        Parameters:
        state: 2x1 numpy array, the state coordinates.

        Returns:
        A scalar value representing the distance from the state to the circular constraint boundary.
        """
        distance = jnp.linalg.norm(state[:2] - center) - padding
        return distance - radius

    return circle_constraint


def create_rectangle_constraint(center: np.ndarray, length: float, padding: float):
    """
    Creates a rectangle constraint function based on the given parameters.

    Parameters:
    center: 2x1 numpy array, the center coordinates of the rectangle
    length: scalar, the length of the rectangle.
    padding: scalar, the padding value for the rectangle.

    Returns:
    A rectangle constraint function.
    """
    def rectangle_constraint(state: np.ndarray):
        """
        Rectangle constraint function that computes the distance from the state to the rectangular constraint boundary.

        Parameters:
        state: 2x1 numpy array, the state coordinates.

        Returns:
        A scalar value representing the distance from the state to the rectangular constraint boundary.
        """
        # Extend the length of the square by the padding
        length_with_padding = length + 2 * padding

        # Coordinate of bottom left corner of the obstacle
        bottom_left = jnp.array(center - length_with_padding / 2)

        # Returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
        return -jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length_with_padding[0] - state[0],
                                   state[1] - bottom_left[1], bottom_left[1] + length_with_padding[1] - state[1]]))

    return rectangle_constraint


def create_circle_constraint(center: np.ndarray, radius: float, padding: float):
    """
    Creates a circular constraint function based on the given parameters.

    Parameters:
    center: 2x1 numpy array, the center coordinates of the circle
    radius: float, the radius of the circle.
    padding: float, the padding value around the circle.

    Returns:
    A circle constraint function.
    """
    def circle_constraint(state: np.ndarray):
        """
        Circular constraint function that computes the distance from the state to the circular constraint boundary.

        Parameters:
        state: 2x1 numpy array, the state coordinates.

        Returns:
        A scalar value representing the distance from the state to the circular constraint boundary.
        """
        distance = jnp.linalg.norm(state[:2] - center) - padding
        return distance - radius

    return circle_constraint


def create_bounding_box_constraint(center: np.ndarray, length: float, padding: float):
    """
    Creates a bounding box constraint function based on the given parameters.

    Parameters:
    center: 2x1 numpy array, the center coordinates of the bounding box.
    length: float, the length of the bounding box.
    padding: float, the padding value for the bounding box.

    Returns:
    A bounding box constraint function.
    """

    def bounding_box_constraint(state: np.ndarray):
        """
        Bounding box constraint function that computes the distance from the state to the bounding box constraint boundary.

        Parameters:
        state: 2x1 numpy array, the state coordinates.

        Returns:
        A scalar value representing the distance from the state to the bounding box constraint boundary.
        """
        # Retract the length of the square by the padding
        length_with_padding = length - 2 * padding

        # Coordinate of bottom left corner of the obstacle
        bottom_left = jnp.array(center - length_with_padding / 2)

        # Returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
        return jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length_with_padding[0] - state[0],
                                  state[1] - bottom_left[1], bottom_left[1] + length_with_padding[1] - state[1]]))

    return bounding_box_constraint


def create_rectangle_constraint(center: np.ndarray, length: float, padding: float):
    """
    Creates a rectangle constraint function based on the given parameters.

    Parameters:
    center: 2x1 numpy array, the center coordinates of the rectangle
    length: scalar, the length of the rectangle.
    padding: scalar, the padding value for the rectangle.

    Returns:
    A rectangle constraint function.
    """
    def rectangle_constraint(state: np.ndarray):
        """
        Rectangle constraint function that computes the distance from the state to the rectangular constraint boundary.

        Parameters:
        state: 2x1 numpy array, the state coordinates.

        Returns:
        A scalar value representing the distance from the state to the rectangular constraint boundary.
        """
        # Extend the length of the square by the padding
        length_with_padding = length + 2 * padding

        # Coordinate of bottom left corner of the obstacle
        bottom_left = jnp.array(center - length_with_padding / 2)

        # Returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
        return -jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length_with_padding[0] - state[0],
                                   state[1] - bottom_left[1], bottom_left[1] + length_with_padding[1] - state[1]]))

    return rectangle_constraint


def define_constraint_function(obstacles: dict, padding: float):
    """
    Defines a constraint function l(x) based on obstacles provided.

    Args:
        obstacles: Python dictionary, obstacles with the key being the obstacle type and the value being a dictionary of obstacle parameters.
        padding: Float that inflates the obstacles by a certain amount using Minkowski sum.

    Returns:
        A constraint function based on the provided obstacles for use in Python HJ Reachability package.
        Takes current state as argument and returns a scalar (will be positive if the state is in the constraint set (i.e., outside an obstacle), negative or 0 otherwise).
    """
    constraints_list = []  # list of constraints

    # loop through obstacles dictionary and define constraint set based on obstacle type
    for obstacle_type in obstacles:

        # if obstacle type is circular
        if obstacle_type == "circle":

            # loop through circle dictionary and define constraint set
            for circle in obstacles[obstacle_type]:

                constraint = create_circle_constraint(obstacles[obstacle_type][circle]["center"],
                                                      obstacles[obstacle_type][circle]["radius"], padding)
                constraints_list.append(constraint)

        elif obstacle_type == "bounding_box":

            for bounding_box in obstacles[obstacle_type]:

                constraint = create_bounding_box_constraint(obstacles[obstacle_type][bounding_box]["center"],
                                                            obstacles[obstacle_type][bounding_box]["length"],
                                                            padding)
                constraints_list.append(constraint)

        elif obstacle_type == "rectangle":

            for rectangle in obstacles[obstacle_type]:

                constraint = create_rectangle_constraint(obstacles[obstacle_type][rectangle]["center"],
                                                         obstacles[obstacle_type][rectangle]["length"], padding)
                constraints_list.append(constraint)

        elif obstacle_type == "iteration":
            pass

        else:  # if obstacle type is not supported yet
            error_msg = f"Obstacle type '{obstacle_type}' is not supported yet."
            raise NotImplementedError(error_msg)


    def constraint_function(state: np.ndarray):
        """
        A real-valued function s.t. the zero-superlevel set complement is the failure set - i.e., obstacles.

        Args:
            state: An unbatched (!) state vector, an array of shape `(3,)` containing `[x, y, omega]`.

        Returns:
            A scalar, positive iff the state is not in the failure set, negative or 0 otherwise.
        """
        # initialize numpy array of constraints
        numpy_array_of_constraints = np.array([])

        # loop through list of constraints
        for l in constraints_list:
            # append constraint to numpy array
            numpy_array_of_constraints = jnp.append(numpy_array_of_constraints, l(state))

        # loop through numpy array of constraints and take the piecewise minimum
        return jnp.min(numpy_array_of_constraints)

    return constraint_function


def save_float_to_file(data: float, filename: str):
    """
    Saves a floating-point value to a file. If the file exists, appends the value at the end;
    otherwise, creates a new file and writes the value.

    Parameters:
    data: float, the value to be saved.
    filename: str, the name of the file.

    Returns:
    None
    """
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
    """
    Prints an error message to the console if the state feedback configuration is not properly configured.
    For example, if one of the GLOBAL state config variables is set to something other than True or False.

    Returns:
    None
    """
    print("Error: State feedback not properly configured. Please check config.py file.")
    exit()


def configure_state_feedback_subscriber(self, qos, topic_string: str):
    """
    Assigns the state feedback subscriber based on the configuration in config.py
    to the self.state_sub attribute of the ROS node class instance.

    Parameters:
    self: ROS node class instance.
    qos: Quality of Service settings.
    topic_string: str, the name of the topic.

    Returns:
    state_sub: ROS subscriber object.
    """
    if topic_string == 'gazebo/odom' or topic_string == 'odom':
        state_sub = self.create_subscription(
            Odometry,
            topic_string,
            self.state_sub_callback_odom,
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


def configure_nominal_policy_publisher(self, qos, USE_UNFILTERED_POLICY: bool):
    """
    Configures and creates the nominal policy publisher based on the configuration in config.py.

    Parameters:
    self: ROS node class instance.
    qos: Quality of Service settings.
    USE_UNFILTERED_POLICY: bool, whether to use unfiltered policy or not.

    Returns:
    nom_pol_publisher_: ROS publisher object.
    """
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


def create_nominal_policy_publishing_message(USE_UNFILTERED_POLICY: bool):
    """
    Creates a message about the nominal policy publishing based on the configuration in config.py.

    Parameters:
    USE_UNFILTERED_POLICY: bool, whether to use unfiltered policy or not.

    Returns:
    nominal_policy_message: str, a message about the nominal policy publishing topic.
    """
    if USE_UNFILTERED_POLICY is True:
        nominal_policy_message = 'Publishing nominal control input over topic /cmd_vel.'

    elif USE_UNFILTERED_POLICY is False:
        nominal_policy_message = 'Publishing nominal control input over topic /nom_policy.'
    else:
        print("Error: USE_UNFILTERED_POLICY is not configured correctly. Please check config.py file.")

    return nominal_policy_message


def create_new_obstacle_set(self, obstacles: dict, iteration: int):
    """
    Creates a new obstacle set based on the provided obstacles for the given iteration.

    Parameters:
    self: ROS node class instance.
    obstacles: python dictionary, obstacles with the key being the obstacle type and the value being a dictionary of obstacle parameters.
    iteration: int, the current iteration number.

    Returns:
    None
    """
    print('New obstacle introduced at iteration: ', iteration)

    # redefine the constraint set
    self.constraint_set = define_constraint_function(obstacles, self.obst_padding)


def introduce_obstacle(self, OBSTACLE_LIST: list, OBSTACLE_ITERATION_LIST: list):
    """
    Introduces a new obstacle based on the provided list of obstacles and their corresponding iterations.

    Parameters:
    self: ROS node class instance.
    OBSTACLE_LIST: list, a list of dictionaries containing obstacle information.
    OBSTACLE_ITERATION_LIST: list, a list of integers representing the iteration number for each obstacle.

    Returns:
    None
    """
    for i, iteration in enumerate(OBSTACLE_ITERATION_LIST):
        if self.iteration == iteration:
            create_new_obstacle_set(self, OBSTACLE_LIST[i], iteration)
            break
    else:
        return

    # redefine the obstacle
    self.obstacle_hjr = hj.utils.multivmap(self.constraint_set, jnp.arange(self.grid.ndim))(self.grid.states)
    # redefine the brt function
    brt_fct = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))  # Backwards reachable TUBE!
    # redefine the solver settings
    self.solver_settings = hj.SolverSettings.with_accuracy("high", value_postprocessor=brt_fct(self.obstacle_hjr))


def update_obstacle_set(self, OBSTACLE_LIST: list, OBSTACLE_ITERATION_LIST: list):
    """
    Updates the current obstacle set based on the provided list of obstacles and their corresponding iterations.

    Parameters:
    self: ROS node class instance.
    OBSTACLE_LIST: list, a list of dictionaries containing obstacle information.
    OBSTACLE_ITERATION_LIST: list, a list of integers representing the iteration number for each obstacle.

    Returns:
    None
    """
    for i, iteration in enumerate(OBSTACLE_ITERATION_LIST):
        if self.iteration == iteration:
            create_new_obstacle_set(self, OBSTACLE_LIST[i], iteration)
            break
    else:
        return

    # redefine the obstacle
    self.obstacle = hj.utils.multivmap(self.constraint_set, jnp.arange(self.grid.ndim))(self.grid.states)


def swap_x_y_coordinates(vertices: np.ndarray):
    """
    Swaps the x and y coordinates of the vertices to reflect the grid rotation from Python to Rviz / Gazebo.

    Parameters:
    vertices: numpy array of shape (n, 2), where n is the number of vertices, representing the x, y coordinates.

    Returns:
    vertices: numpy array of shape (n, 2) with swapped x, y coordinates.
    """
    for j in range(len(vertices)):
        temp = vertices[j][0]
        vertices[j][0] = vertices[j][1]
        vertices[j][1] = temp

    return vertices


def generate_circle_vertices(radius: float, num_vertices: int, center: tuple = (0, 0)):
    """
    Generates the vertices of a circle with the given radius and center.

    Parameters:
    radius: float, the radius of the circle.
    num_vertices: int, the number of vertices to generate.
    center: tuple, the center coordinates of the circle (default: (0, 0)).

    Returns:
    vertices: numpy array of shape (num_vertices + 1, 2) containing the x, y coordinates of the circle vertices.
    """
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


def create_shutdown_message():
    """
    Creates a ROS Twist message that commands 0 velocity in all components.

    Returns:
    msg: ROS Twist message with zero linear and angular velocities.
    """
    msg = Twist()
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0

    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0

    return msg


def state_sub_callback_odom(self, msg):
    """
    Callback function for subscribing to the state information from the 'odom' topic or 'gazebo/odom' topic.

    Parameters:
    self: ROS node class instance.
    msg: Odometry message containing the state information.

    Updates the 'self.state' attribute with the current state information.
    """
    # Message to terminal
    self.get_logger().info('Received new state information.')

    # Convert quaternion to euler angle
    (roll, pitch, yaw) = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                               msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

    self.state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])

    print("Current State: ", self.state)


def state_sub_callback_tf_stamped(self, msg):
    """
    Callback function for subscribing to the state information from the 'vicon/turtlebot_1/turtlebot_1' topic.

    Parameters:
    self: ROS node class instance.
    msg: TransformStamped message containing the state information.

    Updates the 'self.state' attribute with the current state information.
    """
    # Message to terminal
    self.get_logger().info('Received new state information from Vicon arena.')

    x = msg.transform.translation.x
    y = msg.transform.translation.y
    qx = msg.transform.rotation.x
    qy = msg.transform.rotation.y
    qz = msg.transform.rotation.z
    qw = msg.transform.rotation.w

    # Update state
    self.state = np.array([x, y, qx, qy, qz, qw])

    print("Current State: ", self.state)


def create_sub_callback(cls: Type, topic_name: str):
    """
    Creates a state subscriber callback function based on the topic name.

    Parameters:
    cls: The class to which the callback function will be added as an attribute.
    topic_name: The topic name to which the callback function will be associated.

    Updates the class attribute with the respective state subscriber callback function.
    """
    if topic_name == 'gazebo/odom' or topic_name == 'odom':
        setattr(cls, 'state_sub_callback_odom', state_sub_callback_odom)
    elif topic_name == 'vicon/turtlebot_1/turtlebot_1':
        setattr(cls, 'state_sub_callback_tf_stamped', state_sub_callback_tf_stamped)
    else:
        state_feedback_config_error()