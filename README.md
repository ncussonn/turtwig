# RefineCBF for Differential-Drive Hardware Application
ROS package for the refineCBF algorithm implementation on a Turtlebot3 as part of Nathan Cusson-Nadeau's master's thesis at UC San Diego.
RefineCBF is an algorithm from the paper [Refining Control Barrier Functions using HJ Reachability](https://arxiv.org/abs/2204.12507) by Sander Tonkens and Sylvia Herbert, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022. It provides a constructive algorithmic approach to create valid control barrier functions (CBFs) using a unification between Hamilton-Jacobi Reachability analysis and CBF theory. Valid CBFs can be used in safety-critical controls to ensure safety during real-time operation of autonomous systems.

## Requirements:

This package relies on the following:

- Ubuntu 20.04 LTS operating system. It is possible this code can run on other ROS2 compatible distributions but may require minor alterations and has not been tested.
- Python 3: `sudo apt install python3.x` (where x is most up-to-date python 3 distribution)
- [Foxy Fitzroy](https://github.com/ros2/ros2/releases) ROS2 distribution with RVIZ2.
- ROS Physics Simulator [Gazebo](http://classic.gazebosim.org/tutorials?tut=ros2_installing&cat=connect_ros) for Foxy distribution.
- `hj_reachability`: Toolbox for computing HJ reachability leveraging `jax`: `pip install --upgrade hj-reachability`.
Requires installing `jax` additionally based on available accelerator support. See [JAX installation instructions](https://github.com/google/jax#installation) for details.
- `cbf_opt`:  Toolbox for constructing CBFs and implementing them in a safety filter (using `cvxpy`). Run `pip install "cbf_opt>=0.6.0"` or install locally using the [Github link](https://github.com/stonkens/cbf_opt) and run `pip install -e .` in DIR to install.
- `refine_cbfs`: Complementary library from SASLab refineCBF research code. Contains code to define a tabular CBF (a CBF defined over a grid) and provides an interface with `hj_reachability` and `cbf_opt` to define its dynamics. See [RefineCBF Repo](https://github.com/UCSD-SASLab/refineCBF.git) for installation.
- (Optional) `ros1_bridge`: if needing to interface with ROS1 nodes, it may be necessary to bridge between ROS1 and ROS2 topics, [this bridging package](https://github.com/ros2/ros1_bridge) allows this. For example, if using the Vicon aerodrome arena at UCSD, the standard package publishes state information over ROS1 topics and thus a bridge must be made between them.
- (Optional) `erl_quadrotor_vicon`: if using the UC San Diego aerodrome arena, this package will be necessary to retrieve Vicon state information. See [ERL Github](https://github.com/ExistentialRobotics/erl_quadrotor_vicon). Do not let the name of the package deceive you, it can retrieve the pose of any kind of robot. This will additonally require the ROS1 distibution [Noetic Ninjemys](http://wiki.ros.org/noetic/Installation/Ubuntu).

## User Guide

Full documentation under construction. But to run minimal example in Gazebo:

Gazebo Terminal:
- `source ~/<refine_cbf_ws>/install/setup.bash`
- `ros2 launch turtlebot3_gazebo refine_cbf_experiment_2x2.launch.py`
  
RVIZ Terminal:
- `source /opt/ros/<ros2_distro>/setup.bash`
- `rviz2`
- Apply the provided rviz config file: `refine_cbf.rviz`

RefineCBF Terminal:
- `cd ~/<refine_cbf_ws>` (Must be run in the root of workspace for relative folder pointers!)
- `ros2 launch refine_cbf refine_cbf_launch.py`

(Under construction) See the [code documentation](https://github.com/ncussonn/turtwig/blob/master/RefineCBF_Documentation.pdf) for full description of package and how to use it.

## Time Saving Aliases for .bashrc

Adding the following aliases to your `.bashrc` file may save time during use. Type the desired following commands in your command line:

- Sourcing Package: `echo "alias launch_rcbf='ros2 launch refine_cbf refine_cbf_launch.py'" >> ~/.bashrc`
- Launching Gazebo Experiment: `echo "alias rcbf_gzb='ros2 launch turtlebot3_gazebo refine_cbf_experiment_2x2.launch.py'" >> ~/.bashrc`
- RefineCBF Experiment Launch: `echo "alias launch_rcbf='ros2 launch refine_cbf refine_cbf_launch.py'" >> ~/.bashrc`
- Running Safety Filter node: `echo "alias rcbf_sf='ros2 run refine_cbf safety_filter'" >> ~/.bashrc`
- Running Nominal Policy node: `echo "alias rcbf_np='ros2 run refine_cbf nominal_policy'" >> ~/.bashrc`
- Running Dynamic Programming node: `echo "alias rcbf_dp='ros2 run refine_cbf dynamic_programming'" >> ~/.bashrc`
- Running Visualization node: `echo "alias rcbf_vz='ros2 run refine_cbf refine_cbf_visualization'" >> ~/.bashrc`
- SSH into tb3 (add your tb3 IP address): `echo "alias tb3_ssh='ssh ubuntu@<tb3_ip_address>'" >> ~/.bashrc`

## Package Namesake
This package is codenamed Turtwig after the turtle Pokemon!

![turtwig](https://github.com/ncussonn/turtwig/assets/48974771/b1acaff7-264f-48dc-8739-0c5085f967a5)



