# Configuration file used to define key parameters for RefineCBF Experimentation

import numpy as np
import jax.numpy as jnp
import refine_cbfs
from refine_cbfs.dynamics import HJControlAffineDynamics
from cbf_opt import ControlAffineDynamics, ControlAffineCBF
from experiment_utils import *

# LIST OF PARAMETERS SOMEONE MIGHT WANT TO CHANGE
# 1. HJ Grid
# 2. Constraint set / Obstacles
# 3. Dynamic Model
# 4. Control Space Parameters
# 5. Disturbance Space
# 6. Initial CBF

## HAMILTON JACOBI REACHABILITY GRID

# Density of grid
# tuple of size equal to state space dimension that defines the number of grid points in each dimension
# For example, a differential drive dynamics robot will have a 3D state space (x, y, theta)
GRID_RESOLUTION = (31, 31, 21)

# Lower and upper bounds of the discretized state space
# numpy array with dimensions equal to state space dimension
# For example, a differential drive dynamics robot will have a 3D state space and the resulting numpy array will be of size (3,)
GRID_LOWER = np.array([0., 0., -np.pi])
GRID_UPPER = np.array([2., 2., np.pi])

# Periodic dimensions
# A single integer or tuple of integers denoting which dimensions are periodic in the case that
# the `boundary_conditions` are not explicitly provided as input.
# For example, a differential drive dynamics robot will have a 3D state space (x, y, theta), where theta is periodic, which is the third dimension (index 2)
PERIODIC_DIMENSIONS = 2

## BYPASS SAFETY FILTER

# If True, the safety filter will be bypassed and the nominal policy will be used
# If False, the safety filter will be used
bypass_safety_filter = False

## CONTROL SPACE

# Control space parameters
# two arrays with size of the control space dimension that define the lower and upper bounds of the control space
# For instance, a differential drive dynamics robot will have a 2D control space (v, omega) and if
# using a Turtlebot3 Burger, the control space will be bounded by v_min = 0.1, v_max = 0.22, omega_max/omega_min = +/- 2.63

U_MIN = np.array([0.1, -2.63])
U_MAX = np.array([0.21, 2.63])

## DYNAMICS

DYNAMICS = DiffDriveDynamics({"dt": 0.05}, test=False)  # dt is an arbitrary value choice, as the dynamics object requires a dt 
                                                       # value for its constructor argument but it is not used for this package

DYNAMICS_JAX_NUMPY = DiffDriveJNPDynamics({"dt": 0.05}, test=False) # dt is an arbitrary value choice, as the dynamics object requires a dt 
                                                                   # value for its constructor argument but it is not used for this package

DYNAMICS_HAMILTON_JACOBI_REACHABILITY = HJControlAffineDynamics(DYNAMICS_JAX_NUMPY, control_space=hj.sets.Box(jnp.array(U_MIN), jnp.array(U_MAX)))

## CONTROL BARRIER FUNCTION (CBF)

# Gamma value for the CBF
GAMMA = 1.0

# Scalar multiplier for the CBF
SCALAR = 1.0

# Initial CBF Parameters
RADIUS_CBF = 0.15 # radius of the circular CBF
CENTER_CBF = np.array([0.5, 0.5]) # center of the circular CBF

CBF = DiffDriveCBF(DYNAMICS, {"center": CENTER_CBF, "r": RADIUS_CBF, "scalar": SCALAR}, test=False)

# constructor parameters: dt = 0.05, test = False
#dyn = DiffDriveDynamics({"dt": dt}, test=False)

# instantiate an objected called dyn_jnp based on the DiffDriveJNPDynamics class
# constructor parameters: dt = 0.05, test = False
#dyn_jnp = DiffDriveJNPDynamics({"dt": dt}, test=False)

## USE SIMULATION FOR STATE

# If True, the state will retrieved from Gazebo simulation
# If False, the state will be retrieved from sensors
use_simulation_for_state = False

# USE VICON ARENA FOR STATE

# If True, the state is intending to be retrieved from Vicon Arena
# If False, the state is not intending to be retrieved from Vicon Arena
use_vicon_arena_for_state = False

# TODO: INSERT OTHER STATE RETRIEVAL METHODS HERE

## CONSTRAINT SET / OBSTACLES

# a dictionary of obstacles with the key being the obstacle type and the value being a dictionary of obstacle parameters
OBSTACLES = {"circle": {"circle_1": {"center": np.array([1.0, 1.0]), "radius": 0.15}},
                 "bounding_box": {"bounding_box_1":{"center": np.array([1.0, 1.0]),"length":np.array([2.0,2.0])},
                 "rectangle": {"center": np.array([0.5, 1.5]),"length": np.array([0.15,0.15])}}}

# padding around the obstacle
# float that inflates the obstacles by a certain amount using Minkoswki sum
# For example, if the maximum radius of a robot is 0.15 m, the padding should be at least 0.15 m
OBSTACLE_PADDING = 0.11

# DYNAMICS MODEL

# Options [x] means current functionality implemented:
#  [x] Differential Drive
#  [ ] Ackermann
#  [x] Dubin's Car
#  [ ] 6 DOF Quadcopter
#  [ ] 3 DOF Quadcopter (Planar)
#  [ ] ...

'''Manually change the dynamic model to use'''
dynamic_model = "diff_drive" # dynamic model to use

# CONTROL SPACE PARAMETERS
if dynamic_model == "diff_drive":

    ''' Manually adjust the control space parameters if using diff_drive dynamics '''
    v_min = 0.11 # minimum linear velocity
    v_max = 0.21 # maximum linear velocity
    omega_min = -2.63 # minimum angular velocity
    omega_max = -omega_min # maximum angular velocity

    umin = np.array([v_min, -omega_min]) # 1x2 array that defines the minimum values the linear and angular velocity can take 
    umax = np.array([v_max, omega_max])  # same as above line but for maximum

elif dynamic_model == "dubins_car":

    omega_min = -np.pi/2 # minimum angular velocity
    omega_max = -omega_min # maximum angular velocity

    umin = np.array([-omega_min]) # 1x1 array that defines the minimum values the angular velocity can take
    umax = np.array([omega_max])  # same as above line but for maximum

else: # if dynamic model is not supported yet
    raise NotImplementedError("Only differential drive dynamics and Dubin's car are currently supported")

