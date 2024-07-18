# Point Cloud Processing Application

## Overview
This repository contains a comprehensive point cloud processing and colorization application designed to handle point cloud data, odometry information, and corresponding image data. It includes functionality for smoothing, pose optimization, and point cloud colorization based on image data. The application leverages PCL (Point Cloud Library) for point cloud operations and supports both automatic and manual pose optimization methods.

## Features
- **Point Cloud Loading**: Load point cloud data from PCD files.
- **Odometry Integration**: Read and apply odometry data to transform point clouds.
- **Image Integration**: Load images and apply them for point cloud colorization.
- **Mask Integration**: Optionally apply segmentation masks to point clouds.
- **MLS Smoothing**: Enable Moving Least Squares smoothing on point clouds.
- **NID Optimization**: Optimize the pose using Normalized Information Distance metrics.
- **Manual Initial Guess**: Manually set initial guesses for pose optimization.
- **View Culling**: Remove occluded points from point clouds based on camera viewpoints.
- **Colorization and Smoothing**: Colorize and smooth point clouds based on image data.
- **Visualization**: Visualize the processed point clouds using PCL visualizer.

## Installation (3 ways)

### 1. Build from source
Ensure that the following dependencies are installed:
- PCL (Point Cloud Library)
- Eigen
- OpenCV
- Boost
- OpenMP

You can install the required dependencies using the following commands (assuming Ubuntu):

```bash
sudo apt-get update
sudo apt-get install libpcl-dev libeigen3-dev libopencv-dev libboost-all-dev
```
* Build
```bash
git clone --recursice https://github.com/ChunLI-666/PointCloudProcessor.git && \
cd PointCloudProcessor && \
mkdir build && cd build && \
cmake .. && make -j$(nproc) 
```

### 2. Install with ros1 docker
Follow instructions from following file to install with docker:
- [docker.md](PointCloudProcessor/doc/docker.md)


### 3. Build from dockerfile
#### Build image:
```bash
docker build -t pcd-process-image --network=host .  
```

#### Run the Docker container:
```bash
docker run -it \
    --workdir=/sandbox/ \
    -v $HOME:/sandbox/ \
    -e HOME=/root/ \
    -e "QT_X11_NO_MITSHM=1" \
    -e DISPLAY=unix$DISPLAY \
    --env="DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name=pcd_process \
    --privileged \
    --network host \
    -v /etc/timezone:/etc/timezone:ro \
    -d custom_ros_image
```

#### Attach to the Running Container:
```bash
xhost +local:docker
docker attach pcd_process
```

#### Inside the Docker Container:
1. Test X11 forwarding:
```bash
roscore
rviz
```

2. Set up proxy if needed:
```bash
export http_proxy=http://localhost:3128
export https_proxy=http://localhost:3128
export ftp_proxy=http://localhost:3128
```

3. Run the point cloud processor (assume the organized input dataset is ready):
```bash
./PointCloudProcessor \
    --point_cloud_path /path-to-dataset/scans-clean-mls.pcd \
    --odometry_path /path-to-dataset/vo_interpolated_odom.txt \
    --images_folder /path-to-dataset/image_balanced_color_auto/ \
    --output_path /path-to-dataset/test03/ \
    --enableMLS 0 \
    --enableNIDOptimize 0 \
    --enableInitialGuessManual 1
```

1 PIXEL / - n DISTANCE + n normal
maintain 3d width of crack

colorization image should look good