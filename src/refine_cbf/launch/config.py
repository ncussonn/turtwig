# Configuration file used to define key parameters for RefineCBF Experimentation

import numpy as np
import jax.numpy as jnp
import refine_cbfs
from refine_cbfs.dynamics import HJControlAffineDynamics
from cbf_opt import ControlAffineDynamics, ControlAffineCBF
from .experiment_utils import *

# Type of system being controlled: 'DIFF_DRIVE' or 'QUADROTOR'
# NOTE: Unused right now
SYSTEM_TYPE = 'DIFF_DRIVE'

# Save location of experiment data (written to in safety_filter.py)
DATA_FILENAME = '/home/nate/turtwig_ws/log/test_dataset.txt'
DATA_FILENAME_NOMINAL_POLICY = '/home/nate/turtwig_ws/log/test_dataset_nominal_policy.txt'

# Save location of iteration step data (written to in dynamic_programming.py)
ITERATION_STEP_FILENAME = '/home/nate/turtwig_ws/log/test_iteration_step.txt'

# Hardware Experiment
# If True, transformstamped to odom node will run to allow communication between state feedback in Vicon arena and Rviz (trajectory would not show otherwise)
HARDWARE_EXPERIMENT = False

# State Feedback Topic Name (Gazebo: 'gazebo/odom', Turtlebot3: 'odom', VICON: 'vicon/turtlebot/turtlebot')
STATE_FEEDBACK_TOPIC = 'gazebo/odom'
# STATE_FEEDBACK_TOPIC = 'odom'
# STATE_FEEDBACK_TOPIC = 'vicon/turtlebot_1/turtlebot_1'

# Use unfiltered policy (i.e. only nominal/safety agnostic control applied): True or False
# used in the refine_cbf_launch.py and nominal_policy.py
# If True, this will publish the /cmd_vel topic straight from the nominal policy node instead of the safety filter node
# Run data will not be saved to a the typical data file if this is True (will be found in DATA_FILENAME_NOMINAL_POLICY). TODO: Make it so this isn't the case.
USE_UNFILTERED_POLICY = False

# Use a manually controller for the nominal policy: True or False
USE_MANUAL_CONTROLLER = False

# Refine the CBF: True or False
# If this is False, the final converged CBF will be used in the safety filter and not refined
USE_REFINECBF = True

# Refine CBF Iteration Time Step (dt)
TIME_STEP = 0.15 # default is 0.15

# Initial State / Pose
INITIAL_STATE = np.array([0.5, 1.0, 0])

# Safety Filter ROS Node Timer
SAFETY_FILTER_TIMER_SECONDS = 0.033
SAFETY_FILTER_QOS_DEPTH = 10

# Nominal Policy ROS Node Timer
NOMINAL_POLICY_TIMER_SECONDS = 0.033
NOMINAL_POLICY_QOS_DEPTH = 10

## NOMINAL POLICY TABLE
# Insert the filename of the nominal policy table numpy file, that was precomputed.
NOMINAL_POLICY_FILENAME = '/home/nate/refineCBF/experiment/data_files/2 by 2 Grid/nominal_policy_table_2x2_61_61_61_grid_goal_1pt5x_1pt0y_reduced_omega_bound.npy'
#NOMINAL_POLICY_FILENAME = '/home/nate/refineCBF/experiment/data_files/2 by 2 Grid/nominal_policy_table_2x2_61_61_61_grid_goal_1pt5x_1y.npy'

## HAMILTON JACOBI REACHABILITY GRID

# Density of grid
# tuple of size equal to state space dimension that defines the number of grid points in each dimension
# For example, a differential drive dynamics robot will have a 3D state space (x, y, theta)
# IMPORTANT NOTE: This resolution must be the same as the resolution used to generate the nominal policy table (if using a nominal policy table)
GRID_RESOLUTION = (61, 61, 61)

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


# TODO: DELETE THIS?
GRID = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lo=GRID_LOWER, hi=GRID_UPPER), GRID_RESOLUTION, periodic_dims=PERIODIC_DIMENSIONS)


## CONTROL SPACE

# Control space parameters
# two arrays with size of the control space dimension that define the lower and upper bounds of the control space
# For instance, a differential drive dynamics robot will have a 2D control space (v, omega) and if
# using a Turtlebot3 Burger, the control space will be bounded by v_min = 0.1, v_max = 0.22, omega_max/omega_min = +/- 2.63

#U_MIN = np.array([0.1, -2.63])
#U_MAX = np.array([0.21, 2.63])

U_MIN = np.array([0.1, -1.3])
U_MAX = np.array([0.21, 1.3])

## DISTURBANCE SPACE

# Disturbance space parameters
# two arrays with size of the disturbance space dimension that define the lower and upper bounds of the disturbance space
# For instance, a differential drive dynamics robot will have a 2D disturbance space (v_disturbance, omega_disturbance) and if
# using a Turtlebot3 Burger, the disturbance space will be bounded by v_disturbance_min = -0.1, v_disturbance_max = 0.1, omega_disturbance_max/omega_disturbance_min = +/- 0.1

W_MIN = np.array([-0.1, -0.1])
W_MAX = np.array([0.1, 0.1])

## DYNAMICS

# TODO: Allow dynamics to be changed by keyword argument instead of direct assignment, such as "diff_drive", "dubins_car", etc.
DYNAMICS = DiffDriveDynamics({"dt": 0.05}, test=False)  # dt is an arbitrary value choice, as the dynamics object requires a dt 
                                                        # value for its constructor argument but it is not used for this package

DYNAMICS_JAX_NUMPY = DiffDriveJNPDynamics({"dt": 0.05}, test=False) # dt is an arbitrary value choice, as the dynamics object requires a dt 
                                                                    # value for its constructor argument but it is not used for this package

DYNAMICS_HAMILTON_JACOBI_REACHABILITY = HJControlAffineDynamics(DYNAMICS_JAX_NUMPY, control_space=hj.sets.Box(jnp.array(U_MIN), jnp.array(U_MAX)))

DYNAMICS_HAMILTON_JACOBI_REACHABILITY_WITH_DISTURBANCE = HJControlAffineDynamics(DYNAMICS_JAX_NUMPY, control_space=hj.sets.Box(jnp.array(U_MIN), jnp.array(U_MAX)), disturbance_space=hj.sets.Box(jnp.array(W_MIN), jnp.array(W_MAX)))

## CONTROL BARRIER FUNCTION (CBF)

# Gamma value / discount rate for the CBVF - affects how quickly system can approach boundary of the safe set
# A higher gamma value will make the safety control more conservative, while a lower gamma value will make the safety control more aggressive
# With gamma =0 resulting in least-restrictive control
GAMMA = 0.25 # KEEP GAMMA AT 0.25 FOR ALL EXPERIMENTS

# Scalar multiplier for the CBF - linear multiplier for values of the CBF.
# Should rarely need to change this, but it is here if needed.
CBF_SCALAR = 1.0

# Initial CBF Parameters
RADIUS_CBF = 0.33 # radius of the circular CBF
CENTER_CBF = np.array([0.5, 1.0]) # center of the circular CBF

CBF = DiffDriveCBF(DYNAMICS, {"center": CENTER_CBF, "r": RADIUS_CBF, "scalar": CBF_SCALAR}, test=False)

# CBF Filename
CBF_FILENAME = '/home/nate/thesis/Visualization Code/safety_filter_example_cbf.npy'

## CONSTRAINT SET / OBSTACLES

# Initial set of obstacles
OBSTACLES_1 = {
    "circle": {
        "circle_1": {"center": np.array([1.1, 1.0]), "radius": 0.1},
    },
    "rectangle": {
        "rectangle_1": {"center": np.array([1.5, 0.5]), "length": np.array([0.25, 0.25])},
        "rectangle_2": {"center": np.array([1.5, 1.5]), "length": np.array([0.25, 0.25])},
    },
    "bounding_box": {
        "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
    },
}
# 2nd set of obstacles
OBSTACLES_2  = {
    "circle": {
        "circle_1": {"center": np.array([1.1, 1.0]), "radius": 0.1},
    },
    "rectangle": {
        "rectangle_1": {"center": np.array([1.5, 0.5]), "length": np.array([0.25, 0.25])},
        "rectangle_2": {"center": np.array([1.5, 1.5]), "length": np.array([0.25, 0.25])},
    },
    "bounding_box": {
        "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
    },
}

# 3rd set of obstacles
OBSTACLES_3 = {
    "circle": {
        "circle_1": {"center": np.array([1.1, 1.0]), "radius": 0.1},
    },
    "rectangle": {
        "rectangle_1": {"center": np.array([1.5, 0.5]), "length": np.array([0.25, 0.25])},
        "rectangle_2": {"center": np.array([1.5, 1.5]), "length": np.array([0.25, 0.25])},
    },
    "bounding_box": {
        "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
    },
}

# 4th set of obstacles
OBSTACLES_4 = {
    "circle": {
        "circle_1": {"center": np.array([1.1, 1.0]), "radius": 0.1},
    },
    "rectangle": {
        "rectangle_1": {"center": np.array([1.5, 0.5]), "length": np.array([0.25, 0.25])},
        "rectangle_2": {"center": np.array([1.5, 1.5]), "length": np.array([0.25, 0.25])},
    },
    "bounding_box": {
        "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
    },
}

# 5th set of obstacles
OBSTACLES_5 = {
    "circle": {
        "circle_1": {"center": np.array([1.1, 1.0]), "radius": 0.1},
    },
    "rectangle": {
        "rectangle_1": {"center": np.array([1.5, 0.5]), "length": np.array([0.25, 0.25])},
        "rectangle_2": {"center": np.array([1.5, 1.5]), "length": np.array([0.25, 0.25])},
    },
    "bounding_box": {
        "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
    },
}

# When to update obstacles, size of the list must be n-1, where n is the number of obstacle sets
OBSTACLE_ITERATION_LIST = [10, 20, 30, 40]
    
# define the obstacle dictionary list
# each element in the list is a dictionary of obstacles
OBSTACLE_LIST = [OBSTACLES_1, OBSTACLES_2, OBSTACLES_3, OBSTACLES_4, OBSTACLES_5]

# padding around the obstacle in meters
# float that inflates the obstacles by a certain amount using Minkoswki sum
# For example, if the maximum radius of a robot is 0.15 m, the padding should be at least 0.15 m
OBSTACLE_PADDING = 0.11

# Goal Set Parameters, used in refine_cbf_visualization.py
GOAL_SET_RADIUS = 0.10
GOAL_SET_CENTER = np.array([1.5, 1.0])
GOAL_SET_VERTEX_COUNT = 25 # will be used to generate contour in RVIZ for goal set - higher density will make the contour smoother

# Creating the Initial CBF
state_domain = hj.sets.Box(lo=GRID_LOWER, hi=GRID_UPPER) # defining the state_domain
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, GRID_RESOLUTION, periodic_dims=PERIODIC_DIMENSIONS)
diffdrive_cbf = CBF # instatiate a diffdrive_cbf object with the Differential Drive dynamics object
diffdrive_tabular_cbf = refine_cbfs.TabularControlAffineCBF(DYNAMICS, dict(), grid=grid) # tabularize the cbf so that value can be calculated at each grid point
diffdrive_tabular_cbf.tabularize_cbf(diffdrive_cbf) # tabularize the cbf so that value can be calculated at each grid point
INITIAL_CBF = diffdrive_tabular_cbf.vf_table # initial CBF