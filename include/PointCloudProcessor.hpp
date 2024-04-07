#ifndef POINTCLOUDPROCESSOR_HPP
#define POINTCLOUDPROCESSOR_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <open3d/Open3D.h>
#include <string>
#include <vector>
#include "RGBFrames.hpp"
#include "FrameData.hpp"

class PointCloudProcessor {
public:
    PointCloudProcessor(const std::string& pointCloudPath, const std::string& odometryPath, const std::string& imagesFolder);

    void process();

    // Function to convert a PCL Point Cloud to an Open3D Point Cloud
    std::shared_ptr<open3d::geometry::PointCloud> ConvertPCLToOpen3D(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud) {
        // Create an Open3D PointCloud object
        auto o3d_cloud = std::make_shared<open3d::geometry::PointCloud>();

        // Reserve space for points and colors
        o3d_cloud->points_.reserve(pcl_cloud->size());
        o3d_cloud->colors_.reserve(pcl_cloud->size());

        // Loop through the PCL point cloud and copy over the points and colors
        for (const auto &point : *pcl_cloud) {
            // Add the point coordinates
            o3d_cloud->points_.emplace_back(point.x, point.y, point.z);

            // Add the color, converting from RGB to floating-point representation
            o3d_cloud->colors_.emplace_back(point.r / 255.0, point.g / 255.0, point.b / 255.0);
        }

        return o3d_cloud;
    }

    // Function to convert an Open3D Point Cloud to a PCL Point Cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ConvertOpen3DToPCL (const std::shared_ptr<open3d::geometry::PointCloud> &o3d_cloud) {
        // Create a PCL PointCloud object
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        // Reserve space for points
        pcl_cloud->points.resize(o3d_cloud->points_.size());

        // Loop through the Open3D point cloud and copy over the points and colors
        for (size_t i = 0; i < o3d_cloud->points_.size(); i++) {
            // Add the point coordinates
            pcl_cloud->points[i].x = o3d_cloud->points_[i].x();
            pcl_cloud->points[i].y = o3d_cloud->points_[i].y();
            pcl_cloud->points[i].z = o3d_cloud->points_[i].z();

            // Add the color, converting from floating-point to RGB representation
            pcl_cloud->points[i].r = static_cast<uint8_t>(o3d_cloud->colors_[i].r * 255);
            pcl_cloud->points[i].g = static_cast<uint8_t>(o3d_cloud->colors_[i].g * 255);
            pcl_cloud->points[i].b = static_cast<uint8_t>(o3d_cloud->colors_[i].b * 255);
        }

        return pcl_cloud;
    }

private:
    std::string pointCloudPath;
    std::string odometryPath;
    std::string imagesFolder;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudInCameraCoord;
    
    std::vector<FrameData> frames;

    Eigen::Matrix3d R_lidar2cam; // Rotation matrix from lidar to camera
    Eigen::Vector3d t_lidar2cam; // Translation vector from lidar to camera
    Eigen::Isometry3d T_lidar2cam; // Transformation matrix from lidar to camera

    std::vector<double> K_camera(9, 0.0);
    std::vector<double> D_camera(5, 0.0);

    void loadPointCloud();
    void loadVisualOdometry();
    void loadImages();
    void applyFOVDetectionAndHiddenPointRemoval();
    void colorizePoints();
    void smoothColors();
    void saveColorizedPointCloud(const std::string& outputPath);
    void generateColorMap();
};

#endif // POINTCLOUDPROCESSOR_HPP
