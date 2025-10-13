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


## Access ScanNet Sequences

See https://github.com/ScanNet/ScanNet for details on how to access and interpret the data.
