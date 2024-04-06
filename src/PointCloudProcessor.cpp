#include "PointCloudProcessor.hpp"
#include <pcl/io/pcd_io.h> // For loading point cloud
#include <pcl/filters/voxel_grid.h> // Example for downsampling
#include <iostream>
#include <fstream> // For reading odometry file

PointCloudProcessor::PointCloudProcessor(const std::string& pointCloudPath, const std::string& odometryPath, const std::string& imagesFolder) :
    pointCloudPath(pointCloudPath), odometryPath(odometryPath), imagesFolder(imagesFolder) {
    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
}

void PointCloudProcessor::loadPointCloud() {
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pointCloudPath, *cloud) == -1) {
        throw std::runtime_error("Couldn't read point cloud file.");
    }
    std::cout << "Loaded point cloud with " << cloud->points.size() << " points." << std::endl;
}

void PointCloudProcessor::loadVisualOdometry() {
    std::ifstream voFile(odometryPath);
    std::string line;
    while (getline(voFile, line)) {
        std::istringstream iss(line);
        double timestamp, x, y, z, qw, qx, qy, qz;
        if (!(iss >> timestamp >> x >> y >> z >> qw >> qx >> qy >> qz)) { break; }
        // Store or process this odometry data
        

    }
}


void PointCloudProcessor::loadImages() {
    // Load images from imagesFolder
    // This might involve iterating over files in the folder and loading them as cv::Mat
    std::vector<cv::Mat> images;
    cv::String folderPath(imagesFolder + "/*.jpg"); // Assuming images are in JPEG format
    cv::glob(folderPath, images, false); // Get list of image file paths

    for (const auto& imagePath : images) {
        cv::Mat image = cv::imread(imagePath); // Load image using OpenCV
        if (image.empty()) {
            throw std::runtime_error("Couldn't read image file: " + imagePath);
        }
        // Process the image or store it for later use
    }
}

void PointCloudProcessor::applyFOVDetectionAndHiddenPointRemoval(const FrameData& frame) {
    // Apply FOV detection and hidden point removal
    // This will likely involve transformations based on camera pose and intrinsics
    // and removing points that are not visible in the image

    // Assuming you have the camera pose and intrinsics available
    // You can perform the following steps:

    // 1. Project the 3D points onto the image plane using camera intrinsics and pose
    //    You can use the pcl::transformPointCloud function to transform the point cloud
    //    based on the camera pose.

    // 2. Iterate over the projected points and check if they are within the field of view (FOV)
    //    of the camera. You can define the FOV using the image dimensions and camera intrinsics.
    //    Remove the points that are outside the FOV.

    // 3. Remove the points that are occluded or hidden by other objects in the scene.
    //    This can be done by checking the depth values of the projected points and comparing
    //    them with the depth values of the corresponding pixels in the image.
    //    If the depth value of a point is greater than the depth value of the pixel, it means
    //    the point is occluded or hidden and should be removed.

    // Implement the above steps based on your specific camera model and requirements.

    // Example code to remove points outside a rectangular FOV:
    // Assuming you have the FOV defined by top, bottom, left, and right boundaries
    // and the projected points are stored in a pcl::PointCloud<pcl::PointXYZRGB> object called projectedCloud

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    Pose6D pose6d = getOdom(frame.pose);
    Eigen::Affine3f transform = Eigen::Affine3f::Identity(); // Replace with actual transformation
    transform = pcl::getTransformation(pose6d.x, pose6d.y, pose6d.z, pose6d.roll, pose6d.pitch, pose6d.yaw);
    pcl::transformPointCloud(*cloud, *transformedCloud, transform);
    

    
    // Replace the original point cloud with the filtered point cloud
    cloud = filteredCloud;
}

Pose6D getOdom(const Pose& pose)
{
    auto tx = pose.x;
    auto ty = pose.y;
    auto tz = pose.z;

    double roll, pitch, yaw;
    // geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(pose.qx, quat.qy, quat.qz, quat.qw)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw}; 
} // getOdom

void PointCloudProcessor::colorizePoints() {
    // Colorize points based on the projected image coordinates
}

void PointCloudProcessor::smoothColors() {
    // Smooth the colors of the point cloud
}

void PointCloudProcessor::loadImagesAndOdometry() {
    std::ifstream voFile(odometryPath);
    std::string line;
    while (getline(voFile, line)) {
        std::istringstream iss(line);
        double timestamp, x, y, z, qw, qx, qy, qz;
        if (!(iss >> timestamp >> x >> y >> z >> qw >> qx >> qy >> qz)) {
            break; // Error
        }
        Pose pose{x, y, z, qw, qx, qy, qz};

        // std::string imagePath = findImagePathForTimestamp(timestamp);
        std::string imagePath = imagesFolder + "/image_" + std::to_string(timestamp) + ".jpg";
        if (!imagePath.empty()) {
            frames.emplace_back(imagePath, timestamp, pose);
        }
    }
}

void PointCloudProcessor::process() {
    loadPointCloud();
    // loadVisualOdometry();
    // loadImages();
    loadImagesAndOdometry();

    for(const auto& frame : frames) {
        // Process each frame
        applyFOVDetectionAndHiddenPointRemoval(const FrameData& frame);
        colorizePoints();
        smoothColors();

    }
}

void PointCloudProcessor::saveColorizedPointCloud(const std::string& outputPath) {
    if (pcl::io::savePCDFile<pcl::PointXYZRGB>(outputPath, *cloud) == -1) {
        throw std::runtime_error("Couldn't save colorized point cloud.");
    }
    std::cout << "Saved colorized point cloud." << std::endl;
}


std::string PointCloudProcessor::findImagePathForTimestamp(double timestamp) {
    // Implement logic to match image file names to timestamps.
    // This might involve enumerating files in imagesFolder and finding the closest match.
    // Returning an empty string if no match is found.

    return ""; // Placeholder
}