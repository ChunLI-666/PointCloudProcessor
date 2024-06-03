#ifndef POINTCLOUDPROCESSOR_HPP
#define POINTCLOUDPROCESSOR_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <open3d/Open3D.h>
// #include "open3d/Open3D.h"
#include <string>
#include <vector>
#include "RGBFrames.hpp"
#include "FrameData.hpp"
#include "cloudSmooth.hpp"
#include <tf/transform_datatypes.h>

class PointCloudProcessor
{
public:
    PointCloudProcessor(
        const std::string &pointCloudPath,
        const std::string &odometryPath,
        const std::string &imagesFolder,
        const std::string &maskImageFolder,
        const std::string &outputPath,
        const bool &enableMLS);

    void process();

    // Function to convert a PCL Point Cloud to an Open3D Point Cloud
    std::shared_ptr<open3d::geometry::PointCloud> ConvertPCLToOpen3D(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud)
        {
        // Create an Open3D PointCloud object
        auto o3d_cloud = std::make_shared<open3d::geometry::PointCloud>();

        // Reserve space for points and colors
        o3d_cloud->points_.reserve(pcl_cloud->size());
        o3d_cloud->colors_.reserve(pcl_cloud->size());

        // Loop through the PCL point cloud and copy over the points and colors
        for (const auto &point : *pcl_cloud)
        {
            // Add the point coordinates
            o3d_cloud->points_.emplace_back(point.x, point.y, point.z);

            // Add the color, converting from RGB to floating-point representation
            o3d_cloud->colors_.emplace_back(point.r / 255.0, point.g / 255.0, point.b / 255.0);
        }

        return o3d_cloud;
    }

    // Function to convert an Open3D Point Cloud to a PCL Point Cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ConvertOpen3DToPCL(const std::shared_ptr<open3d::geometry::PointCloud> &o3d_cloud)
    {
        // Create a PCL PointCloud object
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        // Reserve space for points
        pcl_cloud->points.resize(o3d_cloud->points_.size());

        // Loop through the Open3D point cloud and copy over the points and colors
        for (size_t i = 0; i < o3d_cloud->points_.size(); i++)
        {
            // Add the point coordinates
            pcl_cloud->points[i].x = o3d_cloud->points_[i].x();
            pcl_cloud->points[i].y = o3d_cloud->points_[i].y();
            pcl_cloud->points[i].z = o3d_cloud->points_[i].z();

            // Add the color, converting from floating-point to RGB representation
            pcl_cloud->points[i].r = static_cast<uint8_t>(o3d_cloud->colors_[i][0] * 255);
            pcl_cloud->points[i].g = static_cast<uint8_t>(o3d_cloud->colors_[i][1] * 255);
            pcl_cloud->points[i].b = static_cast<uint8_t>(o3d_cloud->colors_[i][2] * 255);
        }

        return pcl_cloud;
    }

    std::shared_ptr<open3d::geometry::PointCloud> ConvertMeshToPointCloud(
        const std::shared_ptr<open3d::geometry::TriangleMesh> &mesh)
    {

        // Create a new point cloud
        auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();

        point_cloud->points_.reserve(mesh->vertices_.size());
        point_cloud->colors_.reserve(mesh->vertices_.size());

        for (const auto &vertex : mesh->vertices_)
        {
            point_cloud->points_.push_back(vertex);
            point_cloud->colors_.push_back(Eigen::Vector3d(0.5, 0.5, 0.5));
        }

        return point_cloud;
    }

    Eigen::Vector2d distort(Eigen::Vector2d point)
    {
        double k1 = D_camera[0];
        double k2 = D_camera[1];
        double k3 = D_camera[4];
        double p1 = D_camera[2];
        double p2 = D_camera[3];

        double x2 = point.x() * point.x();
        double y2 = point.y() * point.y();

        double r2 = x2 + y2;
        double r4 = r2 * r2;
        double r6 = r2 * r4;

        double r_coeff = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        double t_coeff1 = 2.0 * point.x() * point.y();
        double t_coeff2 = r2 + 2.0 * x2;
        double t_coeff3 = r2 + 2.0 * y2;
        double x = r_coeff * point.x() + p1 * t_coeff1 + p2 * t_coeff2;
        double y = r_coeff * point.y() + p1 * t_coeff3 + p2 * t_coeff1;

        return Eigen::Vector2d(x, y);
    }

    // Pose6D getPose6DFromOdom(nav_msgs::Odometry::ConstPtr _odom)
    // {
    //     auto tx = _odom->pose.pose.position.x;
    //     auto ty = _odom->pose.pose.position.y;
    //     auto tz = _odom->pose.pose.position.z;

    //     double roll, pitch, yaw;
    //     geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    //     tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    //     return Pose6D{tx, ty, tz, roll, pitch, yaw};
    // } // getOdom

    Pose6D getPose6DFromOdom(const Pose &pose)
    {
        auto tx = pose.x;
        auto ty = pose.y;
        auto tz = pose.z;

        double roll, pitch, yaw;
        // geometry_msgs::Quaternion quat = pose;
        tf::Matrix3x3(tf::Quaternion(pose.qx, pose.qy, pose.qz, pose.qw)).getRPY(roll, pitch, yaw);

        return Pose6D{tx, ty, tz, roll, pitch, yaw};
    } // getOdom

    Pose getPoseFromOdom(const Pose &pose)
    {
        auto tx = pose.x;
        auto ty = pose.y;
        auto tz = pose.z;

        double roll, pitch, yaw;
        // geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
        // tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

        return Pose{tx, ty, tz, pose.qw, pose.qx, pose.qy, pose.qz};
    } // getOdom

    bool markKeyframe(const FrameData &newFrame, const FrameData *lastFrame, const double distThreshold, const double angThreshold)
    {
        double deltaDistance;
        double deltaAngle;

        if (lastFrame == nullptr)
        {
            std::cout << "First frame is always a keyframe." << std::endl;
            return true;
        }

        const Pose &lastFramePose = lastFrame->pose;
        const Pose &newFramePose = newFrame.pose;

        double dx = newFramePose.x - lastFramePose.x;
        double dy = newFramePose.y - lastFramePose.y;
        double dz = newFramePose.z - lastFramePose.z;
        deltaDistance = std::sqrt(dx * dx + dy * dy + dz * dz);

        Eigen::Quaterniond q1(lastFramePose.qw, lastFramePose.qx, lastFramePose.qy, lastFramePose.qz);
        Eigen::Quaterniond q2(newFramePose.qw, newFramePose.qx, newFramePose.qy, newFramePose.qz);
        Eigen::Quaterniond q_diff = q1.inverse() * q2;

        // Convert quaternion difference to angle in degrees
        deltaAngle = q_diff.angularDistance(q1) * 180.0 / M_PI;

        // Print debug information
        // std::cout << "Frame: " << newFrame.imagePath << std::endl;
        // std::cout << "Distance to last frame: " << deltaDistance << " (threshold: " << distThreshold << ")" << std::endl;
        // std::cout << "Angle to last frame: " << deltaAngle << " (threshold: " << angThreshold << ")" << std::endl;

        if (deltaDistance >= distThreshold)
        {
            std::cout << "Selecting frame as keyframe.\n"
                      << std::endl;
            return true;
        }
        else
            // std::cout << "Frame not selected as keyframe.\n" << std::endl;
            return false;
    }

private:
    std::string pointCloudPath;
    std::string odometryPath;
    std::string imagesFolder;
    std::string maskImageFolder;
    std::string outputPath;
    bool enableMLS;
    bool enableNIDOptimize;
    MLSParameters mlsParams;

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudInCameraCoord;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudInWorldWithRGB;
    pcl::PointCloud<PointXYZRGBMask>::Ptr cloudInWorldWithRGBandMask;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudInWorldWithMaskandMappedColor;

    std::vector<FrameData> frames;

    Eigen::Matrix3d R_lidar2cam;   // Rotation matrix from lidar to camera
    Eigen::Vector3d t_lidar2cam;   // Translation vector from lidar to camera
    Eigen::Isometry3d T_lidar2cam; // Transformation matrix from lidar to camera

    std::vector<double> K_camera{9, 0.0};
    std::vector<double> D_camera{5, 0.0};

    void loadPointCloud();
    // void loadVisualOdometry();
    // void loadImages();
    void applyFOVDetectionAndHiddenPointRemoval(const FrameData &frame);
    void generateColorMap(const FrameData &frame,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_color);

    void generateSegmentMap(const FrameData &frame,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_color,
                            pcl::PointCloud<PointXYZRGBMask>::Ptr &pc_color_segmented);

    void generateSegmentMapWithColor(pcl::PointCloud<PointXYZRGBMask>::Ptr &pc_color_segmented,
                                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_color);

    void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);
    // Pose6D getPose6DFromOdom(const Pose &pose);
    // Pose getPoseFromOdom(const Pose &pose);
    void loadImagesAndOdometry();
    // void colorizePoints();
    // void smoothColors();
    void saveColorizedPointCloud();
    // void generateColorMap();
};

#endif // POINTCLOUDPROCESSOR_HPP
