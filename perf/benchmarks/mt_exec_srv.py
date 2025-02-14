import time
import rclpy

import numpy as np

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty as EmptySrv


class ServiceNode(Node):
    def __init__(self):
        super().__init__('srv_node')

        self.depth = 1
        self.number = 8
        self.delay = 1 / 50
        self.frequency = 1 / 100

        self.subs = []
        for i in range(self.number):
            self.subs.append(self.create_service(EmptyMsg, f'topic_{i}', self.subscribe(i), self.depth, callback_group=MutuallyExclusiveCallbackGroup()))

    def subscribe(self, index):
        def closure(msg):
            xxx = time.time()
            timestamp = round(np.subtract(time.time(), xxx), 6)
            self.get_logger().info('>>> srv_{:} @ {:.6f}'.format(index, timestamp))
            time.sleep(self.delay)
        return closure


if __name__ == '__main__':
    rclpy.init()

    node = ServiceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        node.get_logger().info('Beginning service, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
