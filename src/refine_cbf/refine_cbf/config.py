# Configuration file used to define key parameters for RefineCBF Experimentation

import numpy as np
import jax.numpy as jnp

# HAMILTON JACOBI REACHABILITY GRID

# Defining HJ Grid

# Lower and upper bounds of the discretized state space
grid_lower = jnp.array([0., 0., -jnp.pi])
grid_upper = jnp.array([2., 2., jnp.pi])

grid_resolution = (31, 31, 21) # number of grid points in each dimension

# DYNAMIC MODEL

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


# OBSTACLE TYPE

# Options [x] means current functionality implemented:
#  [x] Rectangular
#  [x] Circular
#  [x] Elliptical
#  [ ] Bounding Box
#  [ ] Line
#  [ ] Point
#  [ ] Rectangular Prism
#  [ ] Cylinder
#  [ ] Sphere (could remove circular?)
#  [ ] Ellipsoid (could remove elliptical?)
#  [ ] Union
#  [ ] Polygonal
#  [ ] ...

# Defining Constraint Set
# returns a function that is a constraint set based on the obstacle type
# TODO: make this more generalizable
def define_constraint_set(obstacle_type, **kwargs):

    """
    Defines the constraint set based on the obstacle type

    Args:
        obstacle_type : String that defines the type of obstacle to use

    Returns:
        A function that is a constraint set based on the obstacle type
    """

    if obstacle_type == "rectangular":

        def constraint_set(state):

            """
            A real-valued function s.t. the zero-superlevel set is the safe set

            Args:
                state : An unbatched (!) state vector, an array of shape `(3,)` containing `[x, y, omega]`.

            Returns:
                A scalar, positive iff the state is in the safe set, negative or 0 otherwise.
            """

            # EDIT THESE PARAMETERS
            # rectangular obstacle parameters
            obstacle_center = np.array([5.0, 5.0]) # rectangluar obstacle center
            obstacle_length = np.array([2.0, 2.0]) # length & width of rectangular obstacle

            # obstacle padding
            obstacle_padding = np.array([0.25, 0.25]) # padding around the obstacle

            # coordinate of bottom left corner of the obstacle
            bottom_left = jnp.array(obstacle_center - obstacle_length / 2 - obstacle_padding)
            # redefine obstacle_length (1x2 array) as length (2x2 square)
            length = obstacle_length + 2 * obstacle_padding
            # returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
            return -jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length[0] - state[0], 
                                    state[1] - bottom_left[1], bottom_left[1] + length[1] - state[1]]))
        
    elif obstacle_type == "circular":

        def constraint_set(state):

            """
            A real-valued function s.t. the zero-superlevel set is the safe set

            Args:
                state : An unbatched (!) state vector, an array of shape `(3,)` containing `[x, y, omega]`.

            Returns:
                A scalar, positive iff the state is in the safe set, negative or 0 otherwise.
            """

            # EDIT THESE PARAMETERS
            # circular obstacle parameters
            obstacle_center = np.array([1, 1])
            obstacle_radius = 0.15

            # obstacle padding
            obstacle_padding = 0.11

            # distance of the state from the center of the obstacle + obstacle padding
            distance = jnp.linalg.norm(state[:2] - obstacle_center) - obstacle_padding

            # returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
            return distance - obstacle_radius
        
    elif obstacle_type == "elliptical":

        def constraint_set(state):

            """
            A real-valued function s.t. the zero-superlevel set is the safe set

            Args:
                state : An unbatched (!) state vector, an array of shape `(3,)` containing `[x, y, omega]`.

            Returns:
                A scalar, positive iff the state is in the safe set, negative or 0 otherwise.
            """

            # EDIT THESE PARAMETERS
            obstacle_center = np.array([5.0, 5.0])
            major_axis = 2.0
            minor_axis = 1.0

            # obstacle padding
            obstacle_padding = 0.25

            # distance of the state from the center of the obstacle
            distance = jnp.linalg.norm(state[:2] - obstacle_center) - obstacle_padding
            # returns a scalar (will be positive if the state is in the safe set, negative or 0 otherwise)
            return distance - (major_axis * minor_axis) / jnp.sqrt((minor_axis * jnp.cos(state[2]))**2 + (major_axis * jnp.sin(state[2]))**2)
        
    elif obstacle_type == "union":

        def constraint_set(state):
            """A real-valued function s.t. the zero-superlevel set is the safe set

            Args:
                state : An unbatched (!) state vector, an array of shape `(3,)` containing `[x, y, omega]`.

            Returns:
                A scalar, positive iff the state is in the safe set
            """
            # minkowski sum inflation
            buffer = 0.110 # (distance in meters around obstacle)
            
            # square perimeter
            bottom_left = jnp.array([0.1+buffer,0.1+buffer])
            length = np.array([1.8-2*buffer,1.8-2*buffer])
            
            radius = 0.15
            
            ##circular obstacle
            distance = jnp.sqrt((state[0] - 1)**2 + (state[1] - 1)**2) - buffer
            
            circle_constraint = distance - radius
            
            # circle
            #constraint_set = circle_constraint
            
            # rectangle
            #constraint_set = -jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length[0] - state[0], 
            #                            state[1] - bottom_left[1], bottom_left[1] + length[1] - state[1]]))

            perimeter_constraint = jnp.min(jnp.array([state[0] - bottom_left[0], bottom_left[0] + length[0] - state[0], 
                                        state[1] - bottom_left[1], bottom_left[1] + length[1] - state[1]]))

            constraint_set = jnp.min(jnp.array([circle_constraint, perimeter_constraint]))
            
            return constraint_set
        
    else: # if obstacle type is not supported yet
        raise NotImplementedError("Obstacle type is not supported yet.")

    return constraint_set