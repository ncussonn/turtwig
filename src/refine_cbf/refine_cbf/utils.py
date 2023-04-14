# utils file for the refine_cbf experiment

import sys; sys.version
# add refine_cbf folder to system path
sys.path.insert(0, '/home/nate/refineCBF')
sys.path.insert(0, '/home/nate/refineCBF/experiment')  # shouldn't need to do this for refine_cbf package, need to find workaround


# Experiment specific packages
import hj_reachability as hj
import refine_cbfs
from refine_cbfs.dynamics import HJControlAffineDynamics
from cbf_opt import ControlAffineDynamics, ControlAffineCBF
import numpy as np
import jax.numpy as jnp
import nominal_hjr_control


# define the obstacle center and length
obstacle_center = np.array([5.0, 5.0]) # obstacle center is at (5,5) meters
obstacle_length = np.array([2.0, 2.0]) # length of obstacle is 2 meters, width of obstacle is 2 meters

# Dynamics
# class for the dynamics of the differential drive robot
# inheritance: ControlAffineDynamics class from the cbf_opt package
class DiffDriveDynamics(ControlAffineDynamics):
    STATES = ['X', 'Y', 'THETA'] # state vector of f(x,u): x
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

    # h(x) (Can also be viewed as V(x) since this is a CBVF)
    # class method: value function, accepts a numpy array of the state vector and float time
    def vf(self, state, time=0.0):
        # returns alpha*(r^2 - (x_state-x_center)^2 - (y_state - y_circle)^2)
        return self.scalar * (self.r ** 2 - (state[..., 0] - self.center[0]) ** 2 - (state[..., 1] - self.center[1]) ** 2)

    # del_h(x)Can also be viewed as del_V(x) since this is a CBVF)
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

# defining function constraint_set
def constraint_set(state):
    """A real-valued function s.t. the zero-superlevel set is the safe set

    Args:
        state : An unbatched (!) state vector, an array of shape `(4,)` containing `[y, v_y, phi, omega]`.

    Returns:
        A scalar, positive iff the state is in the safe set
    """
    # coordinate of bottom left corner of the obstacle
    bottom_left = jnp.array(obstacle_center - obstacle_length / 2)
    # redefine obstacle_length (1x2 array) as length (2x2 square)
    length = obstacle_length
    # returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
    return -jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length[0] - state[0], 
                               state[1] - bottom_left[1], bottom_left[1] + length[1] - state[1]]))

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
   
