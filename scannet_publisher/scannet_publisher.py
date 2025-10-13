import os
import threading

from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class ScanNetPublisher(Node):

    def __init__(self):
        super().__init__('scannet_publisher')

        file_path = self.declare_parameter(
            name='file',
            value=str(),
            descriptor=ParameterDescriptor(read_only=True)
        ).value

        if not file_path:
            raise RuntimeError('File path not provided!')

        if file_path and not os.path.exists(file_path):
            raise RuntimeError(f"File \'{file_path}\' does not exist!")

        self.pub_colour = self.create_publisher(
            CompressedImage, '/camera/color/image_raw/compressed', 1)
        self.pub_depth = self.create_publisher(
            CompressedImage, '/camera/depth/image_raw/compressed', 1)

        self.get_logger().info(f'reading from: {file_path}')

        self.thread_reading = threading.Thread(target=self.read_and_publish, daemon=True)
        self.thread_reading.start()

    def read_and_publish(self):
        pass


def main(args=None):
    rclpy.init(args=args)

    try:
        scannet_publisher = ScanNetPublisher()
        rclpy.spin(scannet_publisher)
    except KeyboardInterrupt:
        pass

    scannet_publisher.destroy_node()

    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
