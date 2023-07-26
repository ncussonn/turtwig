# Defines the nominal controller for the turtlebot3 burger

import numpy as np
from refine_cbf.config import *

class NominalController:
    """
    Nominal Controller Class for the turtlebot3 burger.

    In this class, you define how to compute the nominal control.

    In this example implementation, we use a precomputed nominal policy table for the grid
    resolution and interpolate the nominal policy at the current state.
    """

    def __init__(self, high_level_controller):
        """
        Initializes the Nominal Controller.

        Args:
            high_level_controller: Instance of the high-level controller.
        """
        # Load nominal policy table
        high_level_controller.nominal_policy_table = np.load(NOMINAL_POLICY_FILENAME)

    def compute_nominal_control(self, node):
        """
        Compute the nominal control for the given node.

        Args:
            node: The node instance for which to compute the nominal control.

        Returns:
            The computed nominal policy as a 2D numpy array.
        """
        nominal_policy = node.grid.interpolate(node.nominal_policy_table, node.state)
        nominal_policy = np.reshape(nominal_policy, (1, node.dyn.control_dims))

        return nominal_policy
