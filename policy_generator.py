# Libary Imports
import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, '/home/nate/refineCBF')
from refine_cbfs.dynamics import HJControlAffineDynamics

# Configuration Variables
goal_coordinates = np.array([1.5, 1.5, jnp.pi/2])   # x*, y*, theta*
goal_padding = np.array([.1, .1, jnp.pi])           # padding around the goal state
grid_resolution = {'x':61, 'y':61, 'theta':61}      # number of grid points in each dimension 
                                                    # CRITICALLY IMPORTANT NOTE: This resolution must be the same as the resolution used in config.py.
                                                    # NOTE: This resolution will scale computation time exponentially!

# Advanced Configuration Variables
value_function_time_index = 80 # time index of the value function to use for the nominal policy table. Need to be a time step where value function has converged.

def main():

    # Computing Reachability Value Function to Goal

    # target state
    target = np.array([1.5, 1.5, jnp.pi/2]) # x*, y*, theta*

    # time horizon
    T = -10

    # solve for the optimal control policy for the time horizon seconds using Hamilton Jacobi Reachability
    opt_ctrl = nominal_hjr_control.NominalControlHJNP(dyn_hjr, grid, final_time=T, time_intervals=101, solver_accuracy="low", 
                                                        target=target, padding=jnp.array([.1, .1, jnp.pi]))

    filename = "data_files/2 by 2 Grid/performance_policy_vfs" + file_suffix + ".npy"

    # Solving for the Value Function
    print("Solving for reach value function")
    opt_ctrl.solve()   

    nom_policy = nominal_hjr_control.NominalPolicy(opt_ctrl)
    
    vf_time_index = 80 # time index of the value function to use for the nominal policy table
    
    print("Generating nominal policy table.")
    nominal_policy_table = opt_ctrl.get_nominal_control_table(vf_time_index)
    
    if save_nominal_policy_table is True:
        
        filename = "data_files/2 by 2 Grid/nominal_policy_table" + file_suffix + ".npy"
        np.save(filename, nominal_policy_table)

if __name__ == '__main__':
    main()