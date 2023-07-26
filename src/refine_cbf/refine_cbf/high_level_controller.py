# Defines the nominal controller for the turtlebot3 burger

import numpy as np
from refine_cbf.config import *

# In this class you would define how to compute the nominal control.

# In this example implemenation, we use a precomputed nominal policy table for the grid 
# resolution and interpolate the nominal policy at the current state.

class NominalController():

    def __init__(self, high_level_controller):
        
        # Load nominal policy table
        high_level_controller.nominal_policy_table = np.load(NOMINAL_POLICY_FILENAME)

    def compute_nominal_control(self, node):

        nominal_policy = node.grid.interpolate(node.nominal_policy_table, node.state)
        nominal_policy = np.reshape(nominal_policy, (1, node.dyn.control_dims))

        return nominal_policy