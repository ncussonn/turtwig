# Configuration file used to define key parameters for RefineCBF Experimentation

import numpy as np
import jax.numpy as jnp
import refine_cbfs
from refine_cbfs.dynamics import HJControlAffineDynamics
from cbf_opt import ControlAffineDynamics, ControlAffineCBF
from experiment_utils import *

EXPERIMENT = 1

# Save location of experiment data (written to in safety_filter.py)
#DATA_FILENAME = '/home/nate/turtwig_ws/log/test_dataset.txt'
DATA_FILENAME_NOMINAL_POLICY = '/home/nate/turtwig_ws/log/test_dataset_nominal_policy.txt'

# Experimetn 1.2 - Gazebo:
DATA_FILENAME = '/home/nate/thesis/Datasets/Experiment 1/experiment_1_dataset_1_simulation_2.txt'
#DATA_FILENAME = '/home/nate/thesis/Datasets/Experiment 1/experiment_1_dataset_2_simulation_2.txt'
#DATA_FILENAME = '/home/nate/thesis/Datasets/Experiment 1/experiment_1_dataset_3_simulation_2.txt'

# Experiment 2.2 - Gazebo:
#DATA_FILENAME = '/home/nate/thesis/Datasets/Experiment 2/experiment_2_dataset_1_simulation_2.txt'
#DATA_FILENAME = '/home/nate/thesis/Datasets/Experiment 2/experiment_2_dataset_2_simulation_2.txt'
#DATA_FILENAME = '/home/nate/thesis/Datasets/Experiment 2/experiment_2_dataset_3_simulation_2.txt'

# Experiment 3.2 - Gazebo:
#DATA_FILENAME = '/home/nate/thesis/Datasets/Experiment 3/experiment_3_dataset_1_simulation_2.txt'
#DATA_FILENAME = '/home/nate/thesis/Datasets/Experiment 3/experiment_3_dataset_2_simulation_2.txt'
#DATA_FILENAME = '/home/nate/thesis/Datasets/Experiment 3/experiment_3_dataset_3_simulation_2.txt'


# Save location of iteration step data (written to in dynamic_programming.py)
ITERATION_STEP_FILENAME = '/home/nate/turtwig_ws/log/test_iteration_step.txt'

## EXPERIMENT PARAMETERS

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
USE_UNFILTERED_POLICY = False

# Use a manually controller for the nominal policy: True or False
USE_MANUAL_CONTROLLER = False

# TODO: Add low-level corrective controller functionality
# Use corrective controller: True or False
# USE_CORRECTIVE_CONTROLLER = False

# Refine the CBF: True or False TODO: Check if this functionality still works
USE_REFINECBF = True

# Refine CBF Iteration Time Step (dt)
TIME_STEP = 0.15 # default is 0.15

# Initial State / Pose
INITIAL_STATE = np.array([0.5, 1.0, -np.pi/2])

# Safety Filter ROS Node Timer (seconds)
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

# Scalar multiplier for the CBF
SCALAR = 1.0

# Initial CBF Parameters
RADIUS_CBF = 0.33 # radius of the circular CBF
CENTER_CBF = np.array([0.5, 1.0]) # center of the circular CBF

CBF = DiffDriveCBF(DYNAMICS, {"center": CENTER_CBF, "r": RADIUS_CBF, "scalar": SCALAR}, test=False)

# CBF Filename
CBF_FILENAME = '/home/nate/refineCBF/experiment/data_files/2 by 2 Grid/precomputed_cbf_2x2_61_61_61_grid.npy'

## CONSTRAINT SET / OBSTACLES

#NOTE: Package can currently only handle up to 5 DISJOINT obstacles and 1 update to the obstacles

if EXPERIMENT == 'TEST':

    OBSTACLES_1 = {
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
    }
    # Test Experiment 1 Obstacles
    OBSTACLES_2 = {
        "bounding_box": {
            "bounding_box_1": {"center": np.array([0.5, 1.0]), "length": np.array([1.0, 2.0])},
        },
    }

    # Test Experiment 1 Obstacles
    OBSTACLES_3 = {
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([1.6, 1.6])},
        },
    }

    # Test Experiment 1 Obstacles
    OBSTACLES_4 = {
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([1.4, 1.4])},
        },
    }

    # Test Experiment 1 Obstacles
    OBSTACLES_5 = {
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([1.2, 1.2])},
        },
    }
    # EXPERIMENT 1
    OBSTACLE_ITERATION_LIST = [60, 70, 80, 90]

elif EXPERIMENT == 1:
    # use experiment 1 obstacle set and iterations
    # EXPERIMENT 1 OBSTACLES
    # Experiment 1 Obstacles
    OBSTACLES_1 = {
        "bounding_box": {
            "bounding_box_1": {"center": np.array([0.5, 1.0]), "length": np.array([1.0, 2.0])},
        },
    }

    # Experiment 1 Obstacles
    OBSTACLES_2 = {
        "bounding_box": {
            "bounding_box_1": {"center": np.array([0.5, 1.0]), "length": np.array([1.0, 2.0])},
        },
    }

    # Experiment 1 Obstacles
    OBSTACLES_3 = {
        "bounding_box": {
            "bounding_box_1": {"center": np.array([0.5, 1.0]), "length": np.array([1.0, 2.0])},
        },
    }

    # Experiment 1 Obstacles
    OBSTACLES_4 = {
        "bounding_box": {
            "bounding_box_1": {"center": np.array([0.5, 1.0]), "length": np.array([1.0, 2.0])},
        },
    }

    # Experiment 1 Obstacles
    OBSTACLES_5 = {
        "bounding_box": {
            "bounding_box_1": {"center": np.array([0.5, 1.0]), "length": np.array([1.0, 2.0])},
        },
    }

    # EXPERIMENT 1
    # arbitrarily large so that it does not happen
    OBSTACLE_ITERATION_LIST = [10000, 100001, 100002, 100003]

elif EXPERIMENT == 2:

    # use experiment 2 obstacle set and iterations
    OBSTACLES_1 = {
            # "circle": {
            #     "circle_1": {"center": np.array([1.5, 0.5]), "radius": 0.5},
            #     "circle_2": {"center": np.array([1.5, 1.5]), "radius": 0.5},
            #     # "circle_3": {"center": np.array([0.5, 1.0]), "radius": 0.25},
            #     # "circle_4": {"center": np.array([2.0, 1.0]), "radius": 0.25},
            #     # "circle_5": {"center": np.array([0.0, 1.0]), "radius": 0.25},
            # },
            # "rectangle": {
            #     "rectangle_1": {"center": np.array([1.0, 1.0]), "length": np.array([0.5, 0.5])},
            #     "rectangle_2": {"center": np.array([1.0, 1.5]), "length": np.array([0.25, 0.5])},
            # },
            "bounding_box": {
                "bounding_box_1": {"center": np.array([0.5, 1.0]), "length": np.array([1.0, 2.0])},
            },
        }

    # introduces the below obstacle dictionary into the environment at the given iteration
    OBSTACLE_2_ITERATION = 10

    OBSTACLES_2 = {
        "circle": {
            "circle_1": {"center": np.array([1.75, 1.75]), "radius": 0.75},
            "circle_2": {"center": np.array([1.75, 0.25]), "radius": 0.75},
            "circle_3": {"center": np.array([1.1, 1.0]), "radius": 0.1},
        },
        # "rectangle": {
        #     "rectangle_1": {"center": np.array([1.0, 0.33]), "length": np.array([0.25, 0.66])},
        #     "rectangle_2": {"center": np.array([1.0, 1.66]), "length": np.array([0.25, 0.66])},
        # },
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
    }

    OBSTACLE_3_ITERATION = 20

    OBSTACLES_3 = {

        "circle": {
            "circle_1": {"center": np.array([2.0, 2.0]), "radius": 1.0},
            "circle_2": {"center": np.array([2.0, 0]), "radius": 1.0},
            "circle_3": {"center": np.array([1.1, 1.0]), "radius": 0.1},
        },
        
        # "rectangle": {
        #     "rectangle_1": {"center": np.array([1.0, 0.33]), "length": np.array([0.25, 0.66])},
        #     "rectangle_2": {"center": np.array([1.0, 1.66]), "length": np.array([0.25, 0.66])},
        # },
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
    }

    OBSTACLE_4_ITERATION = 30

    OBSTACLES_4 = {

        "circle": {
            "circle_1": {"center": np.array([2.0, 2.25]), "radius": 1.0},
            "circle_2": {"center": np.array([2.0, -0.25]), "radius": 1.0},
            "circle_3": {"center": np.array([1.1, 1.0]), "radius": 0.1},
        },
        # "rectangle": {
        #     "rectangle_1": {"center": np.array([1.0, 0.33]), "length": np.array([0.25, 0.66])},
        #     "rectangle_2": {"center": np.array([1.0, 1.66]), "length": np.array([0.25, 0.66])},
        # },
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
    }

    OBSTACLE_5_ITERATION = 40

    OBSTACLES_5 = {

        "circle": {
            "circle_1": {"center": np.array([2.0, 2.75]), "radius": 1.0},
            "circle_2": {"center": np.array([2.0, -0.75]), "radius": 1.0},
            "circle_3": {"center": np.array([1.1, 1.0]), "radius": 0.1},
        },
        # "rectangle": {
        #     "rectangle_1": {"center": np.array([1.0, 0.33]), "length": np.array([0.25, 0.66])},
        #     "rectangle_2": {"center": np.array([1.0, 1.66]), "length": np.array([0.25, 0.66])},
        # },
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
    }

    # each element in the list is a integer representing the iteration at which the obstacle associated obstacle set is introduced
    OBSTACLE_ITERATION_LIST = [10, 20, 30, 40]

elif EXPERIMENT == 3:

    # use experiment 3 obstacle set and iterations
    # a dictionary of obstacles with the key being the obstacle type and the value being a dictionary of obstacle parameters
    # use experiment 2 obstacle set and iterations
    OBSTACLES_1 = {
            # "circle": {
            #     "circle_1": {"center": np.array([1.5, 0.5]), "radius": 0.5},
            #     "circle_2": {"center": np.array([1.5, 1.5]), "radius": 0.5},
            #     # "circle_3": {"center": np.array([0.5, 1.0]), "radius": 0.25},
            #     # "circle_4": {"center": np.array([2.0, 1.0]), "radius": 0.25},
            #     # "circle_5": {"center": np.array([0.0, 1.0]), "radius": 0.25},
            # },
            # "rectangle": {
            #     "rectangle_1": {"center": np.array([1.0, 1.0]), "length": np.array([0.5, 0.5])},
            #     "rectangle_2": {"center": np.array([1.0, 1.5]), "length": np.array([0.25, 0.5])},
            # },
            "bounding_box": {
                "bounding_box_1": {"center": np.array([0.5, 1.0]), "length": np.array([1.0, 2.0])},
            },
        }

    # introduces the below obstacle dictionary into the environment at the given iteration
    OBSTACLE_2_ITERATION = 10

    OBSTACLES_2 = {
        "circle": {
            "circle_1": {"center": np.array([1.75, 1.75]), "radius": 0.75},
            "circle_2": {"center": np.array([1.75, 0.25]), "radius": 0.75},
            "circle_3": {"center": np.array([1.1, 1.0]), "radius": 0.1},
        },
        # "rectangle": {
        #     "rectangle_1": {"center": np.array([1.0, 0.33]), "length": np.array([0.25, 0.66])},
        #     "rectangle_2": {"center": np.array([1.0, 1.66]), "length": np.array([0.25, 0.66])},
        # },
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
    }

    OBSTACLE_3_ITERATION = 20

    OBSTACLES_3 = {

        "circle": {
            "circle_1": {"center": np.array([2.0, 2.0]), "radius": 1.0},
            "circle_2": {"center": np.array([2.0, 0]), "radius": 1.0},
            "circle_3": {"center": np.array([1.1, 1.0]), "radius": 0.1},
        },
        
        # "rectangle": {
        #     "rectangle_1": {"center": np.array([1.0, 0.33]), "length": np.array([0.25, 0.66])},
        #     "rectangle_2": {"center": np.array([1.0, 1.66]), "length": np.array([0.25, 0.66])},
        # },
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
    }

    OBSTACLE_4_ITERATION = 30

    OBSTACLES_4 = {

        "circle": {
            "circle_1": {"center": np.array([2.0, 2.25]), "radius": 1.0},
            "circle_2": {"center": np.array([2.0, -0.25]), "radius": 1.0},
            "circle_3": {"center": np.array([1.1, 1.0]), "radius": 0.1},
        },
        # "rectangle": {
        #     "rectangle_1": {"center": np.array([1.0, 0.33]), "length": np.array([0.25, 0.66])},
        #     "rectangle_2": {"center": np.array([1.0, 1.66]), "length": np.array([0.25, 0.66])},
        # },
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
    }

    OBSTACLE_5_ITERATION = 40

    OBSTACLES_5 = {

        "circle": {
            "circle_1": {"center": np.array([2.0, 2.75]), "radius": 1.0},
            "circle_2": {"center": np.array([2.0, -0.75]), "radius": 1.0},
            "circle_3": {"center": np.array([1.1, 1.0]), "radius": 0.1},
        },
        # "rectangle": {
        #     "rectangle_1": {"center": np.array([1.0, 0.33]), "length": np.array([0.25, 0.66])},
        #     "rectangle_2": {"center": np.array([1.0, 1.66]), "length": np.array([0.25, 0.66])},
        # },
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
    }

    # each element in the list is a integer representing the iteration at which the obstacle associated obstacle set is introduced
    OBSTACLE_ITERATION_LIST = [OBSTACLE_2_ITERATION, OBSTACLE_3_ITERATION, OBSTACLE_4_ITERATION, OBSTACLE_5_ITERATION]

else:

    # error, exit program
    print("ERROR: Invalid experiment number")
    exit()
    
# define the obstacle dictionary list
# each element in the list is a dictionary of obstacles
OBSTACLE_LIST = [OBSTACLES_1, OBSTACLES_2, OBSTACLES_3, OBSTACLES_4, OBSTACLES_5]

# padding around the obstacle in meters
# float that inflates the obstacles by a certain amount using Minkoswki sum
# For example, if the maximum radius of a robot is 0.15 m, the padding should be at least 0.15 m
OBSTACLE_PADDING = 0.11 #0.11 is default

# Goal Set Parameters
GOAL_SET_RADIUS = 0.15
GOAL_SET_CENTER = np.array([1.5, 1.0])
GOAL_SET_VERTEX_COUNT = 25

# Creating the Initial CBF
state_domain = hj.sets.Box(lo=GRID_LOWER, hi=GRID_UPPER) # defining the state_domain
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, GRID_RESOLUTION, periodic_dims=PERIODIC_DIMENSIONS)
diffdrive_cbf = CBF # instatiate a diffdrive_cbf object with the Differential Drive dynamics object
diffdrive_tabular_cbf = refine_cbfs.TabularControlAffineCBF(DYNAMICS, dict(), grid=grid) # tabularize the cbf so that value can be calculated at each grid point
diffdrive_tabular_cbf.tabularize_cbf(diffdrive_cbf) # tabularize the cbf so that value can be calculated at each grid point
INITIAL_CBF = diffdrive_tabular_cbf.vf_table # initial CBF


# EXPERIMENT 2
#INITIAL_CBF = np.load()