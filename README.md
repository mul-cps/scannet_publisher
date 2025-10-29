# ROS 2 publisher for ScanNet sequences

## Usage

```bash
ros2 run scannet_publisher scannet_publisher --ros-args -p file:="$SEQUENCE.sens"
```

Parameters:
- `file` (type: `string`): path to the `.sens` file

Topics:
- `/camera/color/image_raw/compressed` (type: `sensor_msgs/msg/CompressedImage`): compressed colour image
- `/camera/depth/image_raw/compressed` (type: `sensor_msgs/msg/CompressedImage`): compressed depth image in 16bit PNG format (1 unit : 1 millimetre)
- `/camera/color/image_raw` (type: `sensor_msgs/msg/Image`): color image (bgr8)
- `/camera/depth/image_raw` (type: `sensor_msgs/msg/Image`): depth image (16UC1)
- `/camera/color/camera_info` (type: `sensor_msgs/msg/CameraInfo`): color intrinsics
- `/camera/depth/camera_info` (type: `sensor_msgs/msg/CameraInfo`): depth intrinsics
- `/tf` (type: `tf2_msgs/msg/TFMessage`): camera poses from extrinsics
- `/clock` (type: `rosgraph_msgs/msg/Clock`): log file time stamp




## Access ScanNet Sequences

See https://github.com/ScanNet/ScanNet for details on how to access and interpret the data.
