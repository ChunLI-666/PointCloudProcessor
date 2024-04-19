#include "PointCloudProcessor.hpp"
#include "cloudSmooth.hpp"
#include <pcl/io/pcd_io.h>          // For loading point cloud
#include <pcl/filters/voxel_grid.h> // Example for downsampling
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <iostream>
#include <fstream> // For reading odometry file

#include <Eigen/Dense> // Add missing include statement for Eigen library
#include <tf/transform_datatypes.h>

PointCloudProcessor::PointCloudProcessor(const std::string &pointCloudPath,
                                         const std::string &odometryPath,
                                         const std::string &imagesFolder,
                                         const std::string &outputPath,
<<<<<<< HEAD
                                         const bool enableMLS)
    : pointCloudPath(pointCloudPath), 
    odometryPath(odometryPath), 
    imagesFolder(imagesFolder), 
    outputPath(outputPath),
    enableMLS(enableMLS)
=======
                                         const bool &enableMLS)
    : pointCloudPath(pointCloudPath),
      odometryPath(odometryPath),
      imagesFolder(imagesFolder),
      outputPath(outputPath),
      enableMLS(enableMLS)
>>>>>>> cf51b1eaca41b518a26adc4aa2f0725b12612160

{
    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloudInWorldWithRGB.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    R_lidar2cam << -0.99993085, -0.00561199, -0.0103344,
        0.01032389, 0.00189784, -0.99994491,
        0.0056313, -0.99998245, -0.00183977;
    t_lidar2cam << 0.071771636420221, -0.04934294727365431, -0.0677501086411397;
    // R_cam2imu;
    // t_cam2imu;
    K_camera = {4818.200388954926, 0.0, 2032.4178620390019,
                0.0, 4819.10345841615, 1535.1895959282901,
                0.0, 0.0, 1.0};
    D_camera = {0.003043514741045163, 0.06634739187544138, -0.000217681797407554, -0.0006654964142658197, 0};

    // Create an instance of the MLSParameters structure to hold the MLS parameters
<<<<<<< HEAD
    // bool enableMLS = false;
    MLSParameters mlsParams;
=======
    // MLSParameters mlsParams;
>>>>>>> cf51b1eaca41b518a26adc4aa2f0725b12612160
    mlsParams.compute_normals = true;
    mlsParams.polynomial_order = 2;
    // mlsParams.search_radius = 0.03;
    mlsParams.search_radius = 0.02;
    mlsParams.sqr_gauss_param = 0.0009;
    mlsParams.num_threads = 12;
    mlsParams.slp_upsampling_radius = 0.05;
    mlsParams.slp_upsampling_stepsize = 0.01;
    mlsParams.rud_point_density = 50;
    // mlsParams.vgd_voxel_size = 0.001; // 0.001
    mlsParams.vgd_voxel_size = 0.005; // 0.001
    mlsParams.vgd_iterations = 4;
    mlsParams.sor_kmean_neighbour = 6;
    mlsParams.sor_std_dev = 0.3;
    mlsParams.upsampling_enum = METHOD_VOXEL_GRID_DILATION;
}

void PointCloudProcessor::loadPointCloud()
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudWithIntensity(new pcl::PointCloud<pcl::PointXYZI>());
    if (enableMLS)
    {
        CloudSmooth cloudSmooth(pointCloudPath);
        cloudSmooth.initialize(mlsParams);
        cloudSmooth.process(cloudWithIntensity);
        pcl::copyPointCloud(*cloudWithIntensity, *cloud);
    }
    else
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pointCloudPath, *cloud) == -1)
        {
            throw std::runtime_error("Couldn't read point cloud file.");
        }
        std::cout << "Loaded point cloud with " << cloud->points.size() << " points." << std::endl;
    }
}

// void PointCloudProcessor::transformCloudToCamera()
// {

//     Eigen::Affine3f transform = Eigen::Affine3f::Identity(); // Replace with actual transformation
//     transform.linear() = R_lidar2cam.cast<float>();
//     transform.translation() = t_lidar2cam.cast<float>();

//     pcl::transformPointCloud(*cloud, *cloudInCameraCoord, transform);

// }

// void PointCloudProcessor::loadVisualOdometry()
// {
//     std::ifstream voFile(odometryPath);
//     std::string line;
//     while (getline(voFile, line))
//     {
//         std::istringstream iss(line);
//         double timestamp, x, y, z, qw, qx, qy, qz;
//         if (!(iss >> timestamp >> x >> y >> z >> qw >> qx >> qy >> qz))
//         {
//             break;
//         }
//         // Store or process this odometry data
//     }
// }

// void PointCloudProcessor::loadImages()
// {
//     // Load images from imagesFolder
//     // This might involve iterating over files in the folder and loading them as cv::Mat
//     std::vector<cv::Mat> images;
//     cv::String folderPath(imagesFolder + "/*.jpg"); // Assuming images are in JPEG format
//     cv::glob(folderPath, images, false);            // Get list of image file paths

//     for (const auto &imagePath : images)
//     {
//         cv::Mat image = cv::imread(imagePath); // Load image using OpenCV
//         if (image.empty())
//         {
//             throw std::runtime_error("Couldn't read image file: " + imagePath);
//         }
//         // Process the image or store it for later use
//     }
// }

/**
 * Applies field of view detection and hidden point removal to the given frame's point cloud.
 * This function transforms the point cloud from the world coordinate system to the camera coordinate system,
 * removes points that are occluded or hidden by other objects in the scene, and applies point cloud colorization
 * by projecting the point cloud onto the image frame and assigning the closest color to each point.
 *
 * @param frame The frame data containing the point cloud and camera pose.
 */
void PointCloudProcessor::applyFOVDetectionAndHiddenPointRemoval(const FrameData &frame)
{
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudInCameraPose(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanInBodyWithRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanInWorldWithRGB(new pcl::PointCloud<pcl::PointXYZRGB>());

    // Pose6D pose6d = getOdom(frame.pose);
    // Eigen::Affine3f t_w2c = Eigen::Affine3f::Identity(); // camera odometry
    // Eigen::Affine3f t_c2w = pcl::getTransformation(pose6d.x, pose6d.y, pose6d.z, pose6d.roll, pose6d.pitch, pose6d.yaw);

    // 1. Transform point cloud from world coordinates to camera pose coordinates
    Pose voPose = getPoseFromOdom(frame.pose);
    Eigen::Isometry3d t_w2c = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d t_c2w = Eigen::Isometry3d::Identity();
    Eigen::Quaterniond q_c2w(voPose.qw, voPose.qx, voPose.qy, voPose.qz); // qw, qx, qy, qz
    Eigen::Vector3d trans_c2w(voPose.x, voPose.y, voPose.z);
    t_c2w.translate(trans_c2w);
    t_c2w.rotate(q_c2w);

    t_w2c = t_c2w.inverse();
    Eigen::Affine3f transformation_w2c = t_w2c.cast<float>();
    Eigen::Affine3f transformation_c2w = t_c2w.cast<float>();
    pcl::transformPointCloud(*cloud, *cloudInCameraPose, transformation_w2c);

    // 2. hidden point removal
    std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud = ConvertPCLToOpen3D(cloudInCameraPose);
    std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud_filtered = std::make_shared<open3d::geometry::PointCloud>();
    // std::shared_ptr<open3d::geometry::TriangleMesh> o3d_cloud_filtered_mesh = std::make_shared<open3d::geometry::TriangleMesh>();

    // Eigen::Vector3d camera_position = {voPose.x, voPose.y, voPose.z};
    Eigen::Vector3d camera_position = {0, 0, 0};
    double radius = 1000.0; // TODO: hardcode

    auto result = o3d_cloud->HiddenPointRemoval(camera_position, radius);
    auto o3d_cloud_filtered_mesh = std::get<0>(result);
    o3d_cloud_filtered = ConvertMeshToPointCloud(o3d_cloud_filtered_mesh);

    // 3. project 3d points to 2d images
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_filtered = ConvertOpen3DToPCL(o3d_cloud_filtered);
    generateColorMap(frame, pcl_cloud_filtered, scanInBodyWithRGB);

    // 4. Save the colorized pointcloud to seperate PCD file
    // visualizePointCloud(pcl_cloud_filtered);

    std::string filteredPointCloudPath = std::string(outputPath + "filtered_pcd/" + std::to_string(frame.imageTimestamp) + ".pcd");
    pcl::PCDWriter pcd_writer;

    if (pcd_writer.writeBinary(filteredPointCloudPath, *scanInBodyWithRGB) == -1)
    {
        throw std::runtime_error("Couldn't save filtered point cloud to PCD file.");
    }
    std::cout << "Filtered point cloud saved to: " << filteredPointCloudPath << ", the point size is " << pcl_cloud_filtered->size() << std::endl;

    // 5. Transforn colored scan into world frame, and combine them into one big colored cloud
    pcl::transformPointCloud(*scanInBodyWithRGB, *scanInWorldWithRGB, transformation_c2w);

    *cloudInWorldWithRGB += *scanInWorldWithRGB;
}

void PointCloudProcessor::generateColorMap(const FrameData &frame,
                                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc,
                                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_color)
{
    // Eigen::Isometry3d T_cl = camera_state.inverse() * lidar_state;
    // Eigen::Matrix3d Rcl = T_cl.rotation();
    // Eigen::Vector3d tcl = T_cl.translation();
    // cv::Mat rgb = cv_bridge::toCvCopy(*msg_rgb, "bgr8")->image;
    if (frame.imagePath == "")
    {
        throw std::runtime_error("Failed to read image from: " + frame.imagePath);
        return;
    }
    std::cout << "Reading image from: " << frame.imagePath << std::endl;
    cv::Mat rgb = cv::imread(frame.imagePath);
    cv::Mat hsv;
    cv::cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);

    // 调整饱和度和亮度
    float saturation_scale = 1.0; // 饱和度增加 0%
    float brightness_scale = 1.2; // 亮度增加 20%
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
        // Eigen::Vector3d point_pc = {pc->points[i].x, pc->points[i].y, pc->points[i].z};
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
                point_rgb.x = point_camera.x();
                point_rgb.y = point_camera.y();
                point_rgb.z = point_camera.z();
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

void PointCloudProcessor::colorizePoints()
{
    // Colorize points based on the projected image coordinates
}

void PointCloudProcessor::smoothColors()
{
    // Smooth the colors of the point cloud
}

void PointCloudProcessor::saveColorizedPointCloud()
{
    if (cloudInWorldWithRGB->size() > 0)
    {
        std::string cloudInWorldWithRGBDir(outputPath + "cloudInWorldWithRGB.pcd");
        pcl::PCDWriter pcd_writer;
        if (pcd_writer.writeBinary(cloudInWorldWithRGBDir, *cloudInWorldWithRGB) == -1)
        {
            throw std::runtime_error("Couldn't save colorized point cloud.");
        }
        else
        {
            cout << "All colored cloud saved to: " << cloudInWorldWithRGBDir << endl;
        }
    }
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
        std::string imagePath = imagesFolder + std::to_string(timestamp) + ".jpg";
        if (!imagePath.empty())
        {
            frames.emplace_back(imagePath, timestamp, pose);
        }
    }
}

void PointCloudProcessor::process()
{
    loadPointCloud();
    loadImagesAndOdometry();

    bool isKeyframe = true;
    // Initialize keyframe identification variables
    FrameData *previousFrame = nullptr;
    const double distThreshold = 1.0; // meter
    const double angThreshold = 25.0; // degree

    for (const auto &frame : frames)
    {
        isKeyframe = markKeyframe(frame, previousFrame, distThreshold, angThreshold);
        if (isKeyframe)
        {
            std::cout << "\n Processing frame: " << frame.imagePath << std::endl;
            // Process each frame
            applyFOVDetectionAndHiddenPointRemoval(frame);
            // colorizePoints();
            // smoothColors();
            isKeyframe = false;
            previousFrame = const_cast<FrameData *>(&frame);
        }
    }
    saveColorizedPointCloud();
}

// std::string PointCloudProcessor::findImagePathForTimestamp(double timestamp)
// {
//     // Implement logic to match image file names to timestamps.
//     // This might involve enumerating files in imagesFolder and finding the closest match.
//     // Returning an empty string if no match is found.

//     return ""; // Placeholder
// }

void PointCloudProcessor::visualizePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    // Create a PCLVisualizer object
    pcl::visualization::PCLVisualizer viewer("PointCloud Viewer");

    // Set the background of the viewer to black
    viewer.setBackgroundColor(0, 0, 0);

    // Add the point cloud to the viewer with some unique ID
    viewer.addPointCloud<pcl::PointXYZRGB>(cloud, "filteredCloud");

    // Add coordinate axes to the viewer
    viewer.addCoordinateSystem(1.0);

    // Set properties of the point cloud
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "filteredCloud");

    // Main visualization loop
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);                                        // Update the viewer
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sleep to reduce CPU usage
    }

    return;
}
