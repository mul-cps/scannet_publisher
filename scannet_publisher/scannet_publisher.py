import os
import threading

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

from .SensorData import SensorData


class ScanNetPublisher(Node):
    def __init__(self):
        super().__init__('scannet_publisher')

        # --- Parameter: path to .sens file ---
        file_path = self.declare_parameter(
            name='file',
            value=str(),
            descriptor=ParameterDescriptor(read_only=True),
        ).value

        if not file_path:
            raise RuntimeError('File path not provided!')
        if not os.path.exists(file_path):
            raise RuntimeError(f"File '{file_path}' does not exist!")

        # --- Publishers ---
        self.pub_color_raw = self.create_publisher(Image, '/camera/color/image_raw', 1)
        self.pub_color_compressed = self.create_publisher(
            CompressedImage, '/camera/color/image_raw/compressed', 1
        )
        self.pub_depth_raw = self.create_publisher(Image, '/camera/depth/image_raw', 1)
        self.pub_depth_compressed = self.create_publisher(
            CompressedImage, '/camera/depth/image_raw/compressed', 1
        )

        # --- Initialize ---
        self.bridge = CvBridge()
        self.get_logger().info(f'Reading from: {file_path}')
        self.data = SensorData(file_path)

        # --- Start thread ---
        self.thread_reading = threading.Thread(target=self.read_and_publish, daemon=True)
        self.thread_reading.start()

    def read_and_publish(self):
        fps = 30
        rate = self.create_rate(fps)

        for i, frame in enumerate(self.data):

            offset = self.get_clock().now().to_msg()

            try:
                # --- COLOR ---
                color = frame.decompress_color(self.data.color_compression_type)
                color_msg = self.bridge.cv2_to_imgmsg(color, encoding='bgr8')
                color_msg.header.stamp = offset
                color_msg.header.frame_id = 'camera_color'
                self.pub_color_raw.publish(color_msg)

                # Compressed color
                success, color_encoded = cv2.imencode('.jpg', color)
                if success:
                    color_comp_msg = CompressedImage()
                    color_comp_msg.header.stamp = offset
                    color_comp_msg.header.frame_id = 'camera_color'
                    color_comp_msg.format = 'jpeg'
                    color_comp_msg.data = color_encoded.tobytes()
                    self.pub_color_compressed.publish(color_comp_msg)

            except Exception as e:
                self.get_logger().warn(f'Failed to publish color frame {i}: {e}')
                continue

            try:
                # --- DEPTH ---
                depth_bytes = frame.decompress_depth(self.data.depth_compression_type)
                depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(
                    self.data.depth_height, self.data.depth_width
                )

                # Raw depth
                depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
                depth_msg.header.stamp = offset
                depth_msg.header.frame_id = 'camera_depth'
                self.pub_depth_raw.publish(depth_msg)

                # Compressed depth
                success, depth_png = cv2.imencode('.png', depth)
                if success:
                    depth_comp_msg = CompressedImage()
                    depth_comp_msg.header.stamp = offset
                    depth_comp_msg.header.frame_id = 'camera_depth'
                    depth_comp_msg.format = '16UC1; png compressed'
                    depth_comp_msg.data = depth_png.tobytes()
                    self.pub_depth_compressed.publish(depth_comp_msg)

            except Exception as e:
                self.get_logger().warn(f'Failed to publish depth frame {i}: {e}')
                continue

            # Log progress
            if i % 50 == 0:
                self.get_logger().info(f'Published frame {i}/{self.data.num_frames}')

            rate.sleep()

        self.get_logger().info('All frames have been published')

    def __del__(self):
        if hasattr(self, 'data'):
            self.data.close()


def main(args=None):
    rclpy.init(args=args)
    scannet_publisher = None
    try:
        scannet_publisher = ScanNetPublisher()
        rclpy.spin(scannet_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        if scannet_publisher is not None:
            scannet_publisher.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
