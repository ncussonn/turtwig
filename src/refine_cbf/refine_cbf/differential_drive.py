# subscribes to controls topic 'controls'

# performs the controls on the physical turtlebot3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class Turtlebot3(Node):
    
        def __init__(self):
            super().__init__('turtlebot3')
    
            # state subscription
            self.state_sub = self.create_subscription(
                Twist,
                'optimal_control',
                self.state_sub_callback,
                10)
    
            # prevent unused variable warnings
            self.state_sub

        # state callback
        def state_sub_callback(self, msg):
            self.get_logger().info('I heard this optimal control: "%s"' % msg)

def main(args=None):

    rclpy.init(args=args)

    turtlebot3 = Turtlebot3()

    rclpy.spin(turtlebot3)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    turtlebot3.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
