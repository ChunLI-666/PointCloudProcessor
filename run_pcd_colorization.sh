#!/bin/bash

# Define variables
root_dir="/sandbox/Documents/zhongnan/fastlio-color/20240715/_2024-07-15-16-25-49_reconstruct"
enableMLS=false
enableNIDOptimize=false
enableInitialGuessManual=true

# Check if Python script exists
if [ ! -f PointCloudProcessor/scripts/make_vo_odom_for_fastlio.py ]; then
    echo "Python script not found: PointCloudProcessor/scripts/make_vo_odom_for_fastlio.py"
    exit 1
fi

# Run Python script to make VO odom for fastlio
python3 PointCloudProcessor/scripts/make_vo_odom_for_fastlio.py \
    --root_dir "$root_dir"

# Run PointCloudProcessor with the specified parameters
PointCloudProcessor/build/PointCloudProcessor \
    --point_cloud_path "$root_dir/scans-crop-mls-cc.pcd" \
    --odometry_path "$root_dir/vo_interpolated_odom.txt" \
    --images_folder "$root_dir/raw_images/" \
    --output_path "$root_dir/" \
    --enableMLS $enableMLS \
    --enableNIDOptimize $enableNIDOptimize \
    --enableInitialGuessManual $enableInitialGuessManual
    --mask_image_folder "$root_dir/maskpredeeplab/"
