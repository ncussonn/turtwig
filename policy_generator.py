# Change These Variables:

import numpy as np
import jax.numpy as jnp

# Configuration Variables
goal_coordinates    = np.array([1.5, 1.5, jnp.pi/2])            # x*, y*, theta*
goal_padding        = np.array([.1, .1, jnp.pi])                # padding around the goal state
grid_lower_bound    = jnp.array([0., 0., -jnp.pi])               # lowest values of the grid
grid_upper_bound    = jnp.array([2., 2., jnp.pi])                # highest values of the grid
grid_resolution_dictionary = {'x':61, 'y':61, 'theta':61}       # number of grid points in each dimension 
                                                                # CRITICALLY IMPORTANT NOTE: This resolution must be the same as the resolution used in config.py.
                                                                # NOTE: This resolution will scale computation time exponentially!
filename = "./nominal_policy_table_2.npy"       # where you want to save the nominal policy table
refineCBF_directory = '/home/<user>/refineCBF'  # directory where refineCBF package from Sander Tonkens is located
vmin = 0.1                                  # minimum linear velocity of turtlebot3 burger
vmax = 0.21                                 # maximum linear velocity of turtlebot3 burger
wmax = 1.3                                  # maximum angular velocity of turtlebot3 burger

# Advanced Configuration Variables
value_function_time_index = 80  # time index of the value function to use for the nominal policy table. Needs to be a time step where value function has converged.
time_horizon = -10              # time horizon of the optimal control policy
dt = 0.05                       # dynamic programming time step (number of time steps will be time_horizon/dt)

### DO NOT CHANGE ANYTHING BELOW THIS LINE ###

# Libary Imports
import sys
import jax.numpy as jnp
import sys
import hj_reachability as hj

sys.path.insert(0, refineCBF_directory)

import refine_cbfs # Sander's refine cbf Python packaged
from refine_cbfs.dynamics import HJControlAffineDynamics # imports the class HJControlAffineDynamics
# allows portability between cbf_opt package and hj_reachability package
from cbf_opt import ControlAffineDynamics, ControlAffineCBF, ControlAffineASIF # import classes from cbf_opt pip Python package
import matplotlib.pyplot as plt
import hj_reachability as hj # Hamilton Jacobi Reachability package
import seaborn as sns
from experiment_wrapper import RolloutTrajectory, TimeSeriesExperiment, StateSpaceExperiment # Sander's experiment wrapper package - contains classes useful for when simulating an experiment
import nominal_hjr_control # python function library - contains 3 classes: NominalControlHJ, NominalControlHJNP, NominalPolicy

class DiffDriveDynamics(ControlAffineDynamics):
    STATES = ['X', 'Y', 'THETA'] # state vector of f(x,u)
    CONTROLS = ['VEL', 'OMEGA']  # control vector of f(x,u)
    
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

# Time Horizon
T = time_horizon
grid_resolution = (grid_resolution_dictionary['x'], grid_resolution_dictionary['y'], grid_resolution_dictionary['theta'])

umin = np.array([vmin, -wmax])  # 1x2 array that defines the minimum values the linear and angular velocity can take 
umax = np.array([vmax, wmax])   # same as above line but for maximum

def main():

    # instantiate an objected called dyn_jnp based on the DiffDriveJNPDynamics class
    # constructor parameters: dt = 0.05, test = False
    dyn_jnp = DiffDriveJNPDynamics({"dt": dt}, test=False)

    dyn_hjr = HJControlAffineDynamics(dyn_jnp, control_space=hj.sets.Box(jnp.array(umin), jnp.array(umax)))

    state_domain = hj.sets.Box(lo=grid_lower_bound, hi=grid_upper_bound)
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, grid_resolution, periodic_dims=2)

    # Computing Reachability Value Function to Goal

    # target state
    target = goal_coordinates

    # solve for the optimal control policy for the time horizon seconds using Hamilton Jacobi Reachability
    opt_ctrl = nominal_hjr_control.NominalControlHJNP(dyn_hjr, grid, final_time=T, time_intervals=101, solver_accuracy="low", 
                                                        target=target, padding=jnp.array([.1, .1, jnp.pi]))

    # Solving for the Value Function
    print("Solving for reach value function")
    opt_ctrl.solve()   

    print("Generating nominal policy table.")
    nominal_policy_table = opt_ctrl.get_nominal_control_table(value_function_time_index)
    
    np.save(filename, nominal_policy_table)

if __name__ == '__main__':
    main()