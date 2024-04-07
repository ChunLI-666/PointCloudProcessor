#include "PointCloudProcessor.hpp"
#include <pcl/io/pcd_io.h>          // For loading point cloud
#include <pcl/filters/voxel_grid.h> // Example for downsampling
#include <iostream>
#include <fstream> // For reading odometry file

#include <Eigen/Dense> // Add missing include statement for Eigen library

PointCloudProcessor::PointCloudProcessor(const std::string &pointCloudPath, const std::string &odometryPath, const std::string &imagesFolder) : pointCloudPath(pointCloudPath), odometryPath(odometryPath), imagesFolder(imagesFolder)
{
    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    R_lidar2cam << -0.99993085, -0.00561199, -0.0103344,
            0.01032389,  0.00189784, -0.99994491,
            0.0056313,  -0.99998245, -0.00183977;
    t_lidar2cam << 0.071771636420221, -0.04934294727365431, -0.0677501086411397;
    K_camera << 4818.200388954926, 0.0, 2032.4178620390019,
                0.0,   4819.10345841615, 1535.1895959282901,
                0.0, 0.0, 1.0;
    D_camera << 0.003043514741045163, 0.06634739187544138, -0.000217681797407554, -0.0006654964142658197, 0;
}

void PointCloudProcessor::loadPointCloud()
{
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pointCloudPath, *cloud) == -1)
    {
        throw std::runtime_error("Couldn't read point cloud file.");
    }
    std::cout << "Loaded point cloud with " << cloud->points.size() << " points." << std::endl;
}

void PointCloudProcessor::transformCloudToCamera()
{

    Eigen::Affine3f transform = Eigen::Affine3f::Identity(); // Replace with actual transformation
    transform.linear() = R_lidar2cam.cast<float>();
    transform.translation() = t_lidar2cam.cast<float>();

    pcl::transformPointCloud(*cloud, *cloudInCameraCoord, transform);

}

void PointCloudProcessor::loadVisualOdometry()
{
    std::ifstream voFile(odometryPath);
    std::string line;
    while (getline(voFile, line))
    {
        std::istringstream iss(line);
        double timestamp, x, y, z, qw, qx, qy, qz;
        if (!(iss >> timestamp >> x >> y >> z >> qw >> qx >> qy >> qz))
        {
            break;
        }
        // Store or process this odometry data
    }
}

void PointCloudProcessor::loadImages()
{
    // Load images from imagesFolder
    // This might involve iterating over files in the folder and loading them as cv::Mat
    std::vector<cv::Mat> images;
    cv::String folderPath(imagesFolder + "/*.jpg"); // Assuming images are in JPEG format
    cv::glob(folderPath, images, false);            // Get list of image file paths

    for (const auto &imagePath : images)
    {
        cv::Mat image = cv::imread(imagePath); // Load image using OpenCV
        if (image.empty())
        {
            throw std::runtime_error("Couldn't read image file: " + imagePath);
        }
        // Process the image or store it for later use
    }
}

void PointCloudProcessor::applyFOVDetectionAndHiddenPointRemoval(const FrameData &frame)
{
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

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudInCameraPose(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudWithRGB(new pcl::PointCloud<pcl::PointXYZRGB>());

    Pose6D pose6d = getOdom(frame.pose);
    Eigen::Affine3f t_w2c = Eigen::Affine3f::Identity(); // camera odometry
    t_c2w = pcl::getTransformation(pose6d.x, pose6d.y, pose6d.z, pose6d.roll, pose6d.pitch, pose6d.yaw);

    pcl::transformPointCloud(*cloud, *cloudInCameraPose, t_w2c);
    

    // 1. hidden point removal
    std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud = ConvertPCLToOpen3D(cloudInCameraPose);
    std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud_filtered = std::make_shared<open3d::geometry::PointCloud>();
    Eigen::Vector3d camera_position = {pose6d.x, pose6d.y, pose6d.z};
    double radius = 1.0; // TODO: hardcode

    //Perform hidden point removal
    auto result = o3d_cloud->HiddenPointRemoval(camera_position, radius);
    o3d_cloud_filtered = std::get<0>(result);

    // 2. project 3d points to 2d images 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_filtered = ConvertOpen3DToPCL(o3d_cloud_filtered);
    
    generateColorMap(frame, pcl_cloud_filtered, cloudWithRGB);
      

    // Replace the original point cloud with the filtered point cloud
    cloud = filteredCloud;
}

void PointCloudProcessor::generateColorMap(const FrameData &frame, 
                                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc,
                                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_color)
{
    // Eigen::Isometry3d T_cl = camera_state.inverse() * lidar_state;
    // Eigen::Matrix3d Rcl = T_cl.rotation();
    // Eigen::Vector3d tcl = T_cl.translation();
    // cv::Mat rgb = cv_bridge::toCvCopy(*msg_rgb, "bgr8")->image;
    if(frame.imagePath == "")
    {
        throw std::runtime_error("Failed to read image from: " + frame.imagePath);
        return;
    }
    cv::Mat rgb = cv::imread(frame.imagePath);
    cv::Mat hsv;
    cv::cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);

    // 调整饱和度和亮度
    float saturation_scale = 1.5; // 饱和度增加 50%
    float brightness_scale = 1.5; // 亮度增加 50%
    for (int y = 0; y < hsv.rows; y++)
    {
        for (int x = 0; x < hsv.cols; x++)
        {
            hsv.at<cv::Vec3b>(y, x)[1] = cv::saturate_cast<uchar>(hsv.at<cv::Vec3b>(y, x)[1] * saturation_scale);
            hsv.at<cv::Vec3b>(y, x)[2] = cv::saturate_cast<uchar>(hsv.at<cv::Vec3b>(y, x)[2] * brightness_scale);
        }
    }

    // 转换回 BGR 色彩空间
    cv::Mat adjusted_image;
    cv::cvtColor(hsv, adjusted_image, cv::COLOR_HSV2BGR);
    rgb = adjusted_image;

    for (int i = 0; i < pc->points.size(); i++)
    {
        Eigen::Vector3d point_camera = {pc->points[i].x, pc->points[i].y, pc->points[i].z};
        // Eigen::Vector3d point_camera = Rcl * point_pc + tcl;
        if (point_camera.z() > 0)
        {
            Eigen::Vector2d point_2d = (point_camera.head<2>() / point_camera.z()).eval();
            Eigen::Vector2d point_2d_dis = distort(point_2d);
            int u = static_cast<int>(K_camera[0] * point_2d_dis.x() + K_camera[2]);
            int v = static_cast<int>(K_camera[4] * point_2d_dis.y() + K_camera[5]);
            if (u >= 0 && u < rgb.cols && v >= 0 && v < rgb.rows)
            {
                pcl::PointXYZRGB point_rgb;
                point_rgb.x = point_pc.x();
                point_rgb.y = point_pc.y();
                point_rgb.z = point_pc.z();
                point_rgb.b = (rgb.at<cv::Vec3b>(v, u)[0]);
                point_rgb.g = (rgb.at<cv::Vec3b>(v, u)[1]);
                point_rgb.r = (rgb.at<cv::Vec3b>(v, u)[2]);
                pc_color->push_back(point_rgb);
            }
        }
    }
}
// void PointCloudProcessor::generateColorMap(sensor_msgs::ImagePtr msg_rgb, Eigen::Isometry3d &camera_state, Eigen::Isometry3d &lidar_state,
//                                            pcl::PointCloud<pcl::PointXYZINormal>::Ptr &pc,
//                                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_color)
// {
//     Eigen::Isometry3d T_cl = camera_state.inverse() * lidar_state;
//     Eigen::Matrix3d Rcl = T_cl.rotation();
//     Eigen::Vector3d tcl = T_cl.translation();
//     cv::Mat rgb = cv_bridge::toCvCopy(*msg_rgb, "bgr8")->image;

//     cv::Mat hsv;
//     cv::cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);

//     // 调整饱和度和亮度
//     float saturation_scale = 1.5; // 饱和度增加 50%
//     float brightness_scale = 1.5; // 亮度增加 50%
//     for (int y = 0; y < hsv.rows; y++)
//     {
//         for (int x = 0; x < hsv.cols; x++)
//         {
//             hsv.at<cv::Vec3b>(y, x)[1] = cv::saturate_cast<uchar>(hsv.at<cv::Vec3b>(y, x)[1] * saturation_scale);
//             hsv.at<cv::Vec3b>(y, x)[2] = cv::saturate_cast<uchar>(hsv.at<cv::Vec3b>(y, x)[2] * brightness_scale);
//         }
//     }

//     // 转换回 BGR 色彩空间
//     cv::Mat adjusted_image;
//     cv::cvtColor(hsv, adjusted_image, cv::COLOR_HSV2BGR);
//     rgb = adjusted_image;

//     for (int i = 0; i < pc->points.size(); i++)
//     {
//         Eigen::Vector3d point_pc = {pc->points[i].x, pc->points[i].y, pc->points[i].z};
//         Eigen::Vector3d point_camera = Rcl * point_pc + tcl;
//         if (point_camera.z() > 0)
//         {
//             Eigen::Vector2d point_2d = (point_camera.head<2>() / point_camera.z()).eval();
//             Eigen::Vector2d point_2d_dis = distort(point_2d);
//             int u = static_cast<int>(K_camera[0] * point_2d_dis.x() + K_camera[2]);
//             int v = static_cast<int>(K_camera[4] * point_2d_dis.y() + K_camera[5]);
//             if (u >= 0 && u < rgb.cols && v >= 0 && v < rgb.rows)
//             {
//                 pcl::PointXYZRGB point_rgb;
//                 point_rgb.x = point_pc.x();
//                 point_rgb.y = point_pc.y();
//                 point_rgb.z = point_pc.z();
//                 point_rgb.b = (rgb.at<cv::Vec3b>(v, u)[0]);
//                 point_rgb.g = (rgb.at<cv::Vec3b>(v, u)[1]);
//                 point_rgb.r = (rgb.at<cv::Vec3b>(v, u)[2]);
//                 pc_color->push_back(point_rgb);
//             }
//         }
//     }
// }

Pose6D getOdom(const Pose &pose)
{
    auto tx = pose.x;
    auto ty = pose.y;
    auto tz = pose.z;

    double roll, pitch, yaw;
    // geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(pose.qx, quat.qy, quat.qz, quat.qw)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw};
} // getOdom

void PointCloudProcessor::colorizePoints()
{
    // Colorize points based on the projected image coordinates
}

void PointCloudProcessor::smoothColors()
{
    // Smooth the colors of the point cloud
}

void PointCloudProcessor::loadImagesAndOdometry()
{
    std::ifstream voFile(odometryPath);
    std::string line;
    while (getline(voFile, line))
    {
        std::istringstream iss(line);
        double timestamp, x, y, z, qw, qx, qy, qz;
        if (!(iss >> timestamp >> x >> y >> z >> qw >> qx >> qy >> qz))
        {
            break; // Error
        }
        Pose pose{x, y, z, qw, qx, qy, qz};

        // std::string imagePath = findImagePathForTimestamp(timestamp);
        std::string imagePath = imagesFolder + "/image_" + std::to_string(timestamp) + ".jpg";
        if (!imagePath.empty())
        {
            frames.emplace_back(imagePath, timestamp, pose);
        }
    }
}

void PointCloudProcessor::process()
{
    loadPointCloud();
    // transformCloudToCamera();
    // loadVisualOdometry();
    // loadImages();
    loadImagesAndOdometry();

    for (const auto &frame : frames)
    {
        // Process each frame
        applyFOVDetectionAndHiddenPointRemoval(const FrameData &frame);
        colorizePoints();
        smoothColors();
    }
}

void PointCloudProcessor::saveColorizedPointCloud(const std::string &outputPath)
{
    if (pcl::io::savePCDFile<pcl::PointXYZRGB>(outputPath, *cloud) == -1)
    {
        throw std::runtime_error("Couldn't save colorized point cloud.");
    }
    std::cout << "Saved colorized point cloud." << std::endl;
}

std::string PointCloudProcessor::findImagePathForTimestamp(double timestamp)
{
    // Implement logic to match image file names to timestamps.
    // This might involve enumerating files in imagesFolder and finding the closest match.
    // Returning an empty string if no match is found.

    return ""; // Placeholder
}