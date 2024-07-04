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

### Build from source
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

### Install with ros1 docker
Follow instructions from following file to install with docker:
- [docker.md](doc/docker.md)


### Build from dockerfile