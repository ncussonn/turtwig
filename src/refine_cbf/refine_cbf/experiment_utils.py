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
from config import *

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, PointField

# Dynamics
# class for the dynamics of the differential drive robot
# inheritance: ControlAffineDynamics class from the cbf_opt package
class DiffDriveDynamics(ControlAffineDynamics):
    STATES   = ['X', 'Y', 'THETA'] # state vector of f(x,u): x
    CONTROLS = ['VEL', 'OMEGA']  # control vector of f(x,u): u
    
    # parameterized constructor: dictionary object params, bool test, **kwargs
    # **kwargs allows a variable number of keyword arguments to be passed to a Python function
    def __init__(self, params, test = False, **kwargs):
        # assign a list with value 2 to the params key "periodic_dims"
        params["periodic_dims"] = [2]
        # pass the dict params, bool test to the base class constructor ControlAffineDynamics
        super().__init__(params, test, **kwargs)

    # class method: accepts a numpy array state and float time
    def open_loop_dynamics(self, state, time=0.0):
        return np.zeros_like(state) # returns an array of zeros with the same shape and type as a given array

    # class method: accepts numpy array state and float time
    def control_matrix(self, state, time=0.0):
        # numpy.repeat(a, repeats, axis = None)
        # repeats the matrix a, by the number of times specified, along the axis specified
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        # defining the B matrix based on dynamics
        B[..., 0, 0] = np.cos(state[..., 2])
        B[..., 1, 0] = np.sin(state[..., 2])
        B[..., 2, 1] = 1
        return B # returns the control matrix B from f(x,u) = f(x) + Bu

    # class method: accepts numpy array of state and float time
    def disturbance_jacobian(self, state, time=0.0):
        # returns an array of zeros equal to the dimension of the state
        return np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)
    
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
        self.center = params["center"]
        self.r = params["r"]
        self.scalar = params["scalar"]
        # call constructor from the super(base) class ControlAffineCBF
        super().__init__(dynamics, params, **kwargs)

    # h(x) (Can also be viewed as a value function since this is a CBVF)
    # class method: value function, accepts a numpy array of the state vector and float time
    # used to warmstart dynamic programming of CBF
    def vf(self, state, time=0.0):
        # returns alpha*(r^2 - (x_state-x_center)^2 - (y_state - y_center)^2)
        return self.scalar * (self.r ** 2 - (state[..., 0] - self.center[0]) ** 2 - (state[..., 1] - self.center[1]) ** 2)

    # del_h(x) Can also be viewed as the gradient of a value function since this is a CBVF
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
        # return the gradient times the constant that was factored out
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

    def append(self, x, y, theta, safety_value, v, omega):
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.theta = np.append(self.theta, theta)
        self.safety_value = np.append(self.safety_value, safety_value)
        self.v = np.append(self.v, v)
        self.omega = np.append(self.omega, omega)

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

    def plot_all(self):
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
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
        plt.tight_layout()
        plt.show()

    # save data to a csv file
    def save_data(self, filename):
        np.savetxt(filename, np.c_[self.x, self.y, self.theta, self.safety_value, self.v, self.omega], delimiter=',')
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
        print("Data loaded from " + filename)
