```bash
echo "deb [trusted=yes] https://raw.githubusercontent.com/mul-cps/scannet_publisher/resolute-lyrical-amd64/ ./" | sudo tee /etc/apt/sources.list.d/mul-cps_scannet_publisher-resolute-lyrical-amd64.list
echo "yaml https://github.com/mul-cps/scannet_publisher/raw/resolute-lyrical-amd64/local.yaml lyrical" | sudo tee /etc/ros/rosdep/sources.list.d/1-mul-cps_scannet_publisher-resolute-lyrical-amd64.list
```
