import os
import threading

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy import clock
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
import tf2_ros

from .SensorData import SensorData


class ScanNetPublisher(Node):
    def __init__(self):
        super().__init__('scannet_publisher')

        file_path = self.declare_parameter(
            name='file',
            value=str(),
            descriptor=ParameterDescriptor(read_only=True),
        ).value

        self.publish_ground_truth = self.declare_parameter(
            name='ground_truth',
            value=True,
            descriptor=ParameterDescriptor(
                read_only=True,
                description='publish the ground truth camera poses',
            ),
        ).value

        if not file_path:
            raise RuntimeError('File path not provided!')
        if not os.path.exists(file_path):
            raise RuntimeError(f"File '{file_path}' does not exist!")

        self.pub_color_raw = self.create_publisher(Image, '/camera/color/image_raw', 1)
        self.pub_color_compressed = self.create_publisher(
            CompressedImage, '/camera/color/image_raw/compressed', 1
        )
        self.pub_depth_raw = self.create_publisher(
            Image, '/camera/depth/image_raw', 1
        )
        self.pub_depth_compressed = self.create_publisher(
            CompressedImage, '/camera/depth/image_raw/compressed', 1
        )
        self.pub_color_info = self.create_publisher(CameraInfo, '/camera/color/camera_info', 1)
        self.pub_depth_info = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 1)

        self.pub_clock = self.create_publisher(Clock, '/clock', 1)

        self.camera_frame_name = 'camera_color'

        self.system_clock = clock.Clock(clock_type=clock.ClockType.SYSTEM_TIME)

        self.bridge = CvBridge()
        self.get_logger().info(f'Reading from: {file_path}')
        self.data = SensorData(file_path)
        if self.publish_ground_truth:
            self.pub_camera_pose = self.create_publisher(PoseStamped, '/camera_pose', 1)
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.K_color = self.data.intrinsic_color[:3, :3]
        self.K_depth = self.data.intrinsic_depth[:3, :3]

        self.thread_reading = threading.Thread(target=self.read_and_publish, daemon=True)
        self.thread_reading.start()

    def read_and_publish(self):
        fps = 30
        rate = self.create_rate(fps)

        for i, frame in enumerate(self.data):
            tnow = self.system_clock.now().to_msg()

            self.pub_clock.publish(Clock(clock=tnow))

            try:
                color = frame.decompress_color(self.data.color_compression_type)
                color_resized = cv2.resize(
                    color,
                    (self.data.depth_width, self.data.depth_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                scale_x = self.data.depth_width / self.data.color_width
                scale_y = self.data.depth_height / self.data.color_height

                K_color_scaled = self.K_color.copy()
                K_color_scaled[0, 0] *= scale_x
                K_color_scaled[1, 1] *= scale_y
                K_color_scaled[0, 2] *= scale_x
                K_color_scaled[1, 2] *= scale_y

                color_msg = self.bridge.cv2_to_imgmsg(color_resized, encoding='bgr8')
                color_msg.header.stamp = tnow
                color_msg.header.frame_id = self.camera_frame_name
                self.pub_color_raw.publish(color_msg)

                color_width_scaled = int(self.data.color_width * scale_x)
                color_height_scaled = int(self.data.color_height * scale_y)

                self.publish_camera_info(
                    color_msg.header,
                    K_color_scaled,
                    color_width_scaled,
                    color_height_scaled,
                    self.pub_color_info,
                )

                success, color_encoded = cv2.imencode('.jpg', color_resized)
                if success:
                    color_comp_msg = CompressedImage()
                    color_comp_msg.header.stamp = tnow
                    color_comp_msg.header.frame_id = self.camera_frame_name
                    color_comp_msg.format = 'jpeg'
                    color_comp_msg.data = color_encoded.tobytes()
                    self.pub_color_compressed.publish(color_comp_msg)

            except Exception as e:
                self.get_logger().warn(f'Failed to publish color frame {i}: {e}')
                continue

            try:
                depth_bytes = frame.decompress_depth(self.data.depth_compression_type)
                depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(
                    self.data.depth_height, self.data.depth_width
                )

                depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
                depth_msg.header.stamp = tnow
                depth_msg.header.frame_id = self.camera_frame_name
                self.pub_depth_raw.publish(depth_msg)

                self.publish_camera_info(
                    depth_msg.header,
                    self.K_depth,
                    self.data.depth_width,
                    self.data.depth_height,
                    self.pub_depth_info,
                )

                success, depth_png = cv2.imencode('.png', depth)
                if success:
                    depth_comp_msg = CompressedImage()
                    depth_comp_msg.header.stamp = tnow
                    depth_comp_msg.header.frame_id = self.camera_frame_name
                    depth_comp_msg.format = '16UC1; png compressed'
                    depth_comp_msg.data = depth_png.tobytes()
                    self.pub_depth_compressed.publish(depth_comp_msg)

            except Exception as e:
                self.get_logger().warn(f'Failed to publish depth camera {i}: {e}')
                continue

            if self.publish_ground_truth:
                self.publish_extrinsics_tf(
                    frame.camera_to_world, 'map', self.camera_frame_name, tnow
                )

            if i % 50 == 0:
                self.get_logger().info(f'Published frame {i}/{self.data.num_frames}')

            rate.sleep()

        self.get_logger().info('All frames have been published')
        rclpy.try_shutdown()

    def publish_camera_info(self, header, K, width, height, topic_pub):
        cam_info = CameraInfo()
        cam_info.header = header
        cam_info.width = width
        cam_info.height = height
        cam_info.distortion_model = 'plumb_bob'
        cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        cam_info.k = K.flatten().tolist()
        cam_info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        cam_info.p = [
            K[0, 0], K[0, 1], K[0, 2], 0.0,
            K[1, 0], K[1, 1], K[1, 2], 0.0,
            K[2, 0], K[2, 1], K[2, 2], 0.0,
        ]
        topic_pub.publish(cam_info)

    def publish_extrinsics_tf(self, T, parent_frame, child_frame, stamp):
        if not np.isfinite(T).all():
            self.get_logger().warn('Invalid transform — skipping TF publish')
            return

        try:
            U, _, Vt = np.linalg.svd(T[:3, :3])
            Rm = np.dot(U, Vt)
            q = R.from_matrix(Rm).as_quat()
        except Exception as e:
            self.get_logger().warn(f'Failed to compute quaternion: {e}')
            return

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        t.transform.translation.x = float(T[0, 3])
        t.transform.translation.y = float(T[1, 3])
        t.transform.translation.z = float(T[2, 3])

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = parent_frame
        pose_msg.pose.position.x = t.transform.translation.x
        pose_msg.pose.position.y = t.transform.translation.y
        pose_msg.pose.position.z = t.transform.translation.z
        pose_msg.pose.orientation = t.transform.rotation

        self.pub_camera_pose.publish(pose_msg)

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
