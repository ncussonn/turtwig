#!/usr/bin/env python
#
# Copyright (c) 2011, Willow Garage, Inc.
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of {copyright_holder} nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Darby Lim

import os
import select
import sys
import rclpy    # ros client library for python
import time

from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

BURGER_MAX_LIN_VEL = 0.21
BURGER_MAX_ANG_VEL = 2.63

WAFFLE_MAX_LIN_VEL = 0.26
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

msg = """
Sending open loop commands inputs.
"""

e = """
Communications Failed
"""

def print_vels(target_linear_velocity, target_angular_velocity):
    print('Current Velocities:\tlinear velocity {0}\t angular velocity {1} '.format(
        target_linear_velocity,
        target_angular_velocity))

def constrain(input_vel, low_bound, high_bound):
    if input_vel < low_bound:
        input_vel = low_bound
    elif input_vel > high_bound:
        input_vel = high_bound
    else:
        input_vel = input_vel

    return input_vel

def invalid_input():
    print("Invalid input. Please try again.")

def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()

    qos = QoSProfile(depth=10)
    node = rclpy.create_node('open_loop_control')
    #pub = node.create_publisher(Twist, 'nom_policy', qos) # if want to use as nominal policy
    pub = node.create_publisher(Twist, 'cmd_vel', qos)

    status = 0
    control_linear_velocity = 0.0
    control_angular_velocity = 0.0
    target_linear_velocity = 0.0
    target_angular_velocity = 0.0

    # desired_linear_velocity = constrain(float(input("Linear Velocity in m/s (max is 0.21): ")),-BURGER_MAX_LIN_VEL,BURGER_MAX_LIN_VEL)
    # desired_angular_velocity = constrain(float(input("Angular Velocity in rad/s (max is 2.63): ")),-BURGER_MAX_ANG_VEL,BURGER_MAX_ANG_VEL)

    try:
        print(msg)

        # repeat forever
        while(1):
            
            # move forward:
            # 1 meter:
            #   0.2 m/s @ t = 5 sec
            # 2 meter:
            #   0.2 m/s @ t = 10 sec

            # rotation:
            # 90 degrees:
            #   1.57 @ t = 1 sec
            # 180 degrees:
            #   1.57 @ t = 2 sec
            # 360 degrees:
            #   1.57 @ t = 4 sec

            # prompt user if they want to generate another cbf
            control_action = input("What control action would you like to take? (f)orward, (b)ackward, (r) right/clockwise, (l) left/counter clockwise, (q)uit: ")
            
            if control_action == 'f' or control_action == 'b':
                distance = float(input("Step size in meters? (1) 1 meter, (2) meters:"))

                if control_action =='f':
                    control_linear_velocity = 0.2
                elif control_action == 'b':
                    control_linear_velocity = -0.2
                else:
                    invalid_input()
                    break

                if distance == 1:
                    duration = 5
                elif distance == 2:
                    duration = 10
                else:
                    invalid_input()
                    break

            elif control_action == 'r' or control_action == 'l':

                angle = float(input("Angle to rotate in degrees? (1) 90 degrees, (2) 180 degrees, (3) 360 degrees:"))

                if control_action =='r':
                    control_angular_velocity = -1.57079632679 
                elif control_action == 'l':
                    control_angular_velocity = 1.57079632679 
                else:
                    invalid_input()
                    break

                if angle == 1:
                    duration = 1
                elif angle == 2:
                    duration = 2
                elif angle == 3:
                    duration = 4
                else:
                    invalid_input()
                    break

            # on quit, stop moving
            elif control_action == 'q':
                control_linear_velocity = 0.0
                control_angular_velocity = 0.0
                break
            
            # # on even counts, only linear velocity
            # if status % 2 == 0:
                
            #     # on second even count, move backward
            #     # if number is divisible by 4, this is every second even count
            #     if status % 4 == 0:

            #         # move backward for 3 seconds
            #         target_linear_velocity = -desired_linear_velocity # m/s
            #         target_angular_velocity = 0.0 # stop rotating                    

            #     # move forward (if we are not on a number divisible by 4, we are on first even count)
            #     else:

            #         # move forward for 3 seconds
            #         target_linear_velocity = desired_linear_velocity # m/s
            #         target_angular_velocity = 0.0 # stop rotating                    

            # # on odd counts, rotate  
            # else:

            #     if status % 3: 
            #         # rotate clockwise 3 seconds
            #         target_linear_velocity = 0.0  # stop moving forward
            #         target_angular_velocity = -desired_angular_velocity # rad/s   

            #     else:
            #         # rotate counterclockwise for 3 seconds
            #         target_linear_velocity = 0.0  # stop moving forward
            #         target_angular_velocity = desired_angular_velocity # rad/s          
            
            # # increment status timer
            # status += 1

            # # printing topic velocities to console
            # print_vels(target_linear_velocity, target_angular_velocity)

            # # every 20 control inputs, remind user what is happening
            # if status == 20:
            #     print(msg)
            #     status = 0

            twist = Twist()

            twist.linear.x = control_linear_velocity
            twist.linear.y = 0.0
            twist.linear.z = 0.0

            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = control_angular_velocity

            pub.publish(twist)
            print("Linear Velocity (m/s): ", twist.linear.x)
            print("Angular Velocity (rad/s)", twist.angular.z)
            print("Duration (sec): ", duration)

            # use command for the desired duration
            time.sleep(duration)

            # stop moving
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0

            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0

            pub.publish(twist)

            # reset velocities
            control_linear_velocity = 0.0
            control_angular_velocity = 0.0

    except Exception as e:
        print(e)

    finally:
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        pub.publish(twist)

        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == '__main__':
    main()
