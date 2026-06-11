# ROS 2 publisher for ScanNet sequences

## Usage

```bash
ros2 run scannet_publisher scannet_publisher --ros-args -p file:="$SEQUENCE.sens"
```

Parameters:
- `file` (type: `string`): path to the `.sens` file
- `ground_truth` (type: `bool`): publish the ground truth camera poses (default: `True`)

Topics:
- `/camera/color/image_raw/compressed` (type: `sensor_msgs/msg/CompressedImage`): compressed colour image
- `/camera/depth/image_raw/compressed` (type: `sensor_msgs/msg/CompressedImage`): compressed depth image in 16bit PNG format (1 unit : 1 millimetre)
- `/camera/color/image_raw` (type: `sensor_msgs/msg/Image`): color image (bgr8)
- `/camera/depth/image_raw` (type: `sensor_msgs/msg/Image`): depth image (16UC1)
- `/camera/color/camera_info` (type: `sensor_msgs/msg/CameraInfo`): color intrinsics
- `/camera/depth/camera_info` (type: `sensor_msgs/msg/CameraInfo`): depth intrinsics
- `/tf` (type: `tf2_msgs/msg/TFMessage`): camera poses from extrinsics (if `ground_truth:=True`)
- `/camera_pose` (type: `geometry_msgs/msg/PoseStamped`): camera poses from extrinsics (if `ground_truth:=True`)
- `/clock` (type: `rosgraph_msgs/msg/Clock`): log file time stamp




## Access ScanNet Sequences

An overview of all ScanNet scenes is available at: https://kaldir.vc.in.tum.de/scannet_browse/scans/scannet/grouped. Note the sequence number in the format `sceneAAAA_BB` and download the `.sens` file:
```sh
SEQUENCE="scene0000_00"
wget https://kaldir.vc.in.tum.de/scannet/v1/scans/$SEQUENCE/$SEQUENCE.sens
```

You have to agree to the [ScanNet Terms of Use](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf) to access and use the data.

See https://github.com/ScanNet/ScanNet for details on how to access and interpret the data.
