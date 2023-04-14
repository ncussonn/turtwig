# Generates state based on control and time between next control input

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class StatePublisher(Node):
    
        def __init__(self):
            super().__init__('state_estimator')

            # state subscription
            self.state_sub = self.create_subscription(
                Twist,
                'optimal_control',
                self.opt_ctrl_sub_callback,
                10)

            # prevent unused variable warnings
            self.state_sub
            
            self.publisher_ = self.create_publisher(Twist, 'state_topic', 10)
            timer_period = 0.5  # seconds
            self.timer = self.create_timer(timer_period, self.timer_callback)

        # state callback
        def opt_ctrl_sub_callback(self, msg):
            self.get_logger().info('Received new control input: "%s"' % msg)
    
        def timer_callback(self):
            # create the state message
            msg = Twist()
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing pose: "%s"' % msg)

def main():

    print("Starting state estimator node...")

    rclpy.init()

    state_publisher = StatePublisher()

    rclpy.spin(state_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    state_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
