```bash
echo "deb [trusted=yes] https://raw.githubusercontent.com/mul-cps/scannet_publisher/noble-jazzy-amd64/ ./" | sudo tee /etc/apt/sources.list.d/mul-cps_scannet_publisher-noble-jazzy-amd64.list
echo "yaml https://github.com/mul-cps/scannet_publisher/raw/noble-jazzy-amd64/local.yaml jazzy" | sudo tee /etc/ros/rosdep/sources.list.d/1-mul-cps_scannet_publisher-noble-jazzy-amd64.list
```
