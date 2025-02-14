import time
import rclpy

import numpy as np

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty as EmptySrv


class ClientNode(Node):
    def __init__(self):
        super().__init__('clt_node')

        self.depth = 1
        self.number = 8
        self.delay = 1 / 50
        self.frequency = 1 / 100

        self.clts = []
        self.triggers = []
        for i in range(self.number):
            self.clts.append(self.create_client(EmptySrv, f'topic_{i}', self.depth, callback_group=MutuallyExclusiveCallbackGroup()))
            self.triggers.append(self.create_timer(self.frequency, self.publish(i), callback_group=ReentrantCallbackGroup()))

    def request(self, index):
        def closure():
            xxx = time.time()
            self.clts[index].call(EmptyMsg()) # call in blocking mode
            timestamp = round(np.subtract(time.time(), xxx), 6)
            self.get_logger().info('>>> clt_{:} @ {:.6f}'.format(index, timestamp))
            time.sleep(self.delay)
        return closure


if __name__ == '__main__':
    rclpy.init()

    node = ClientNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        node.get_logger().info('Beginning client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
