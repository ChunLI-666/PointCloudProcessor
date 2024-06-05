#include "PointCloudProcessor.hpp"
#include "cloudSmooth.hpp"
#include <pcl/io/pcd_io.h>          // For loading point cloud
#include <pcl/filters/voxel_grid.h> // Example for downsampling
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <iostream>
#include <fstream> // For reading odometry file
#include <filesystem>
#include <Eigen/Dense> // Add missing include statement for Eigen library
#include <tf/transform_datatypes.h>

#include "calibrate.cpp"
#include <vlcal/calib/view_culling.hpp>
#include <camera/create_camera.hpp>

PointCloudProcessor::PointCloudProcessor(const std::string &pointCloudPath,
                                         const std::string &odometryPath,
                                         const std::string &imagesFolder,
                                         const std::string &maskImageFolder,
                                         const std::string &outputPath,
                                         const bool &enableMLS,
                                         const bool &enableNIDOptimize)
    : pointCloudPath(pointCloudPath),
      odometryPath(odometryPath),
      imagesFolder(imagesFolder),
      maskImageFolder(maskImageFolder),
      outputPath(outputPath),
      enableMLS(enableMLS),
      enableNIDOptimize(enableNIDOptimize)

{
    cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    cloudInWorldWithRGB.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloudInWorldWithRGBandMask.reset(new pcl::PointCloud<PointXYZRGBMask>());
    cloudInWorldWithMaskandMappedColor.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

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
    // bool enableMLS = false;
    // MLSParameters mlsParams;
    mlsParams.compute_normals = true;
    mlsParams.polynomial_order = 2;
    // mlsParams.search_radius = 0.03;
    mlsParams.search_radius = 0.02;
    mlsParams.sqr_gauss_param = 0.0009;
    mlsParams.num_threads = 24;
    mlsParams.slp_upsampling_radius = 0.05;
    mlsParams.slp_upsampling_stepsize = 0.01;
    mlsParams.rud_point_density = 50;
    // mlsParams.vgd_voxel_size = 0.001; // 0.001
    mlsParams.vgd_voxel_size = 0.005; // 0.001
    mlsParams.vgd_iterations = 3;
    // mlsParams.upsampling_enum = METHOD_VOXEL_GRID_DILATION;
    mlsParams.upsampling_enum = METHOD_VOXEL_GRID_DILATION;

    mlsParams.sor_kmean_neighbour = 60;
    mlsParams.sor_std_dev = 0.5;

    // Check if maskImageFolder was provided; if not, set enableMaksSegmentation to false
    enableMaskSegmentation = !maskImageFolder.empty();
}

void PointCloudProcessor::loadPointCloud()
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudWithIntensity(new pcl::PointCloud<pcl::PointXYZINormal>());
    if (enableMLS)
    {
        CloudSmooth cloudSmooth(pointCloudPath);
        cloudSmooth.initialize(mlsParams);
        cloudSmooth.process(cloudWithIntensity);
        pcl::copyPointCloud(*cloudWithIntensity, *cloud);
    }
    else
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(pointCloudPath, *cloud) == -1)
        {
            throw std::runtime_error("Couldn't read point cloud file.");
        }
        std::cout << "Loaded point cloud with " << cloud->points.size() << " points." << std::endl;
    }
}

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
    // 0. Init objects
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudInCameraPose(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudInCameraPose(new pcl::PointCloud<pcl::PointXYZI>());

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanInBodyWithRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<PointXYZRGBMask>::Ptr scanInBodyWithRGBandMask(new pcl::PointCloud<PointXYZRGBMask>());

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanInWorldWithRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<PointXYZRGBMask>::Ptr scanInWorldWithRGBandMask(new pcl::PointCloud<PointXYZRGBMask>());

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

    // ////// 2. hidden point removal via open3d
    // std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud = ConvertPCLToOpen3D(cloudInCameraPose);
    // std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud_filtered = std::make_shared<open3d::geometry::PointCloud>();
    // //// std::shared_ptr<open3d::geometry::TriangleMesh> o3d_cloud_filtered_mesh = std::make_shared<open3d::geometry::TriangleMesh>();

    // //// Eigen::Vector3d camera_position = {voPose.x, voPose.y, voPose.z};
    // Eigen::Vector3d camera_position = {0, 0, 0};
    // double radius = 1000.0; // TODO: hardcode

    // auto result = o3d_cloud->HiddenPointRemoval(camera_position, radius);
    // auto o3d_cloud_filtered_mesh = std::get<0>(result);
    // o3d_cloud_filtered = ConvertMeshToPointCloud(o3d_cloud_filtered_mesh);
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_filtered = ConvertOpen3DToPCL(o3d_cloud_filtered);

    ////// 2.1 hidden point removal via NID and view_culling.cpp
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::copyPointCloud(*cloudInCameraPose, *pcl_cloud_filtered);

    // 3. project 3d points to 2d images
    const std::string camera_model = "pinhole";
    std::vector<double> K_camera_coefficients = {K_camera[0], K_camera[4], K_camera[2], K_camera[5]};

    // 3.1(optional): use NID metrics to optimize pose
    Eigen::Affine3f transformation_c2w_optimized = transformation_c2w;
    if (enableNIDOptimize)
    {
        vlcal::VisualLiDARCalibration calib(camera_model, K_camera_coefficients, D_camera, frame);
        calib.calibrate(cloudInCameraPose, pcl_cloud_filtered);
        Eigen::Isometry3d T_camera_lidar_optimized;
        T_camera_lidar_optimized = calib.getOptimizedPose();

        // Update camera pose with optimized camera-lidar extrinsic
        Eigen::Isometry3d t_camere_world_optimized = t_c2w * T_camera_lidar_optimized;
        Eigen::Affine3f transformation_c2w_optimized = t_camere_world_optimized.cast<float>();
    }
    else
    {
        camera::GenericCameraBase::ConstPtr proj = camera::create_camera(camera_model, K_camera_coefficients, D_camera);

        vlcal::ViewCullingParams view_culling_params;
        // view_culling_params.enable_depth_buffer_culling = !params.disable_z_buffer_culling;
        std::cout << "before view_culling!" << std::endl;
        vlcal::ViewCulling view_culling(proj, {4096, 3000}, view_culling_params); // TODO: hardcode
        pcl::PointCloud<pcl::PointXYZI>::Ptr culled_points = view_culling.cull(pcl_cloud_filtered, Eigen::Isometry3d::Identity());

        pcl_cloud_filtered->clear();
        for (const auto &point : culled_points->points)
        {
            pcl::PointXYZI transformed_point;
            transformed_point.x = point.x;
            transformed_point.y = point.y;
            transformed_point.z = point.z;
            transformed_point.intensity = point.intensity;

            pcl_cloud_filtered->push_back(transformed_point);
        }
    }

    generateColorMap(frame, pcl_cloud_filtered, scanInBodyWithRGB);

    if (enableMaskSegmentation)
    {
        // 4. project 3d points to 2d segmentation mask images
        generateSegmentMap(frame, scanInBodyWithRGB, scanInBodyWithRGBandMask);

        // save the segmented point cloud
        std::string filteredPointCloudPath = std::string(outputPath + "filtered_pcd/" + std::to_string(frame.imageTimestamp) + ".pcd");
        pcl::PCDWriter pcd_writer;

        if (pcd_writer.writeBinary(filteredPointCloudPath, *scanInBodyWithRGBandMask) == -1)
        {
            throw std::runtime_error("Couldn't save filtered point cloud to PCD file.");
        }
        std::cout << "Filtered point cloud saved to: " << filteredPointCloudPath << ", the point size is " << pcl_cloud_filtered->size() << std::endl;

        // 5. Transforn colored scan into world frame, and combine them into one big colored cloud
        if (enableNIDOptimize)
        {
            pcl::transformPointCloud(*scanInBodyWithRGBandMask, *scanInWorldWithRGBandMask, transformation_c2w_optimized);
        }
        else
        {
            pcl::transformPointCloud(*scanInBodyWithRGBandMask, *scanInWorldWithRGBandMask, transformation_c2w);
        }

        *cloudInWorldWithRGBandMask += *scanInWorldWithRGBandMask;
    }
    else
    {
        // 4. Save the colorized pointcloud to seperate PCD file
        // visualizePointCloud(pcl_cloud_filtered);

        std::string filteredPointCloudPath = std::string(outputPath + "filtered_pcd/" + std::to_string(frame.imageTimestamp) + ".pcd");
        pcl::PCDWriter pcd_writer;

        if (pcd_writer.writeBinary(filteredPointCloudPath, *scanInBodyWithRGB) == -1)
        {
            throw std::runtime_error("Couldn't save filtered point cloud to PCD file.");
        }
        std::cout << "Filtered point cloud saved to: " << filteredPointCloudPath << ", the point size is " << scanInBodyWithRGB->size() << std::endl;

        // 5. Transforn colored scan into world frame, and combine them into one big colored cloud
        if (enableNIDOptimize)
        {
            pcl::transformPointCloud(*scanInBodyWithRGB, *scanInWorldWithRGB, transformation_c2w_optimized);
        }
        else
        {
            pcl::transformPointCloud(*scanInBodyWithRGB, *scanInWorldWithRGB, transformation_c2w);
        }

        *cloudInWorldWithRGB += *scanInWorldWithRGB;
    }
}

void PointCloudProcessor::generateColorMap(const FrameData &frame,
                                           pcl::PointCloud<pcl::PointXYZI>::Ptr &pc,
                                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_color)
{
    std::cout << "Reading image from: " << frame.imagePath << std::endl;
    cv::Mat rgb = cv::imread(frame.imagePath);
    if (rgb.empty())
    {
        throw std::runtime_error("Failed to read image from: " + frame.imagePath);
        return;
    }
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

void PointCloudProcessor::generateSegmentMap(const FrameData &frame,
                                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_color,
                                             pcl::PointCloud<PointXYZRGBMask>::Ptr &pc_color_segmented)
{

    std::cout << "Reading segment mask image from: " << frame.maskImagePath << std::endl;
    cv::Mat grayImg = cv::imread(frame.maskImagePath, cv::IMREAD_GRAYSCALE);
    if (grayImg.empty())
    {
        // throw std::runtime_error("Failed to read image from: " + frame.maskImagePath);
        std::cout << "Failed to read image from: " << frame.maskImagePath << std::endl;
        return;
    }

    for (int i = 0; i < pc_color->points.size(); i++)
    {
        // Eigen::Vector3d point_pc = {pc->points[i].x, pc->points[i].y, pc->points[i].z};
        Eigen::Vector3d point_camera = {pc_color->points[i].x, pc_color->points[i].y, pc_color->points[i].z};
        // Eigen::Vector3d point_camera = Rcl * point_pc + tcl;
        if (point_camera.z() > 0)
        {
            Eigen::Vector2d point_2d = (point_camera.head<2>() / point_camera.z()).eval();
            Eigen::Vector2d point_2d_dis = distort(point_2d);
            int u = static_cast<int>(K_camera[0] * point_2d_dis.x() + K_camera[2]);
            int v = static_cast<int>(K_camera[4] * point_2d_dis.y() + K_camera[5]);
            if (u >= 0 && u < grayImg.cols && v >= 0 && v < grayImg.rows)
            {
                PointXYZRGBMask point_rgb_segmented;
                point_rgb_segmented.x = point_camera.x();
                point_rgb_segmented.y = point_camera.y();
                point_rgb_segmented.z = point_camera.z();
                point_rgb_segmented.b = pc_color->points[i].b;
                point_rgb_segmented.g = pc_color->points[i].g;
                point_rgb_segmented.r = pc_color->points[i].r;
                point_rgb_segmented.segmentMask = grayImg.at<uchar>(v, u);

                pc_color_segmented->push_back(point_rgb_segmented);
            }
        }
    }
}

void PointCloudProcessor::generateSegmentMapWithColor(pcl::PointCloud<PointXYZRGBMask>::Ptr &pc_color_segmented,
                                                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_color)
{
    unsigned char r = 0, g = 255, b = 0; // Initialize to middle of the first color segment for safety

    // Jet colormap
    const int colormapLength = 4;
    float colormap[colormapLength][3] = {
        {0, 0, 0.5},
        {0, 0.5, 1},
        {0.5, 1, 0.5},
        {1, 0.5, 0}};

    for (size_t i = 0; i < pc_color_segmented->points.size(); i++)
    {
        pcl::PointXYZRGB point_rgb;
        point_rgb.x = pc_color_segmented->points[i].x;
        point_rgb.y = pc_color_segmented->points[i].y;
        point_rgb.z = pc_color_segmented->points[i].z;

        // Map the segmentMask value to RGB using Jet colormap
        float value = static_cast<float>(pc_color_segmented->points[i].segmentMask) / 255.0f; // Assuming segmentMask is normalized between 0 and 1
        value = std::min(std::max(value, 0.0f), 1.0f);                                        // Clamp value to range [0, 1] to avoid out-of-bounds access

        for (int j = 0; j < colormapLength - 1; j++)
        {
            float start = static_cast<float>(j) / (colormapLength - 1);
            float end = static_cast<float>(j + 1) / (colormapLength - 1);

            if (value >= start && value < end)
            {
                float ratio = (value - start) / (end - start);
                r = static_cast<uint8_t>(255 * ((colormap[j + 1][0] - colormap[j][0]) * ratio + colormap[j][0]));
                g = static_cast<uint8_t>(255 * ((colormap[j + 1][1] - colormap[j][1]) * ratio + colormap[j][1]));
                b = static_cast<uint8_t>(255 * ((colormap[j + 1][2] - colormap[j][2]) * ratio + colormap[j][2]));
                break;
            }
        }

        point_rgb.r = r;
        point_rgb.g = g;
        point_rgb.b = b;

        // std::cout << "R: " << static_cast<int>(r) << ", G: " << static_cast<int>(g) << ", B: " << static_cast<int>(b) << std::endl;
        pc_color->push_back(point_rgb);
    }
}

// void PointCloudProcessor::colorizePoints()
// {
//     // Colorize points based on the projected image coordinates!maskImageFolder.empty()
// }

// void PointCloudProcessor::smoothColors()
// {
//     // Smooth the colors of the point cloud
// }

void PointCloudProcessor::saveColorizedPointCloud()
{
    if (enableMaskSegmentation)
    {
        if (cloudInWorldWithRGBandMask->size() > 0)
        {
            std::string cloudInWorldWithRGBDir(outputPath + "cloudInWorldWithRGBandMask.pcd");
            pcl::PCDWriter pcd_writer;

            if (pcd_writer.writeBinary(cloudInWorldWithRGBDir, *cloudInWorldWithRGBandMask) == -1)
            {
                throw std::runtime_error("Couldn't save colorized and segment colored point cloud.");
            }
            else
            {
                cout << "All colored and segment colored cloud saved to: " << cloudInWorldWithRGBandMask << endl;
            }

            generateSegmentMapWithColor(cloudInWorldWithRGBandMask, cloudInWorldWithMaskandMappedColor);
            if (cloudInWorldWithMaskandMappedColor->size() > 0)
            {
                std::string cloudInWorldWithMaskandMappedColorDir(outputPath + "cloudInWorldWithMaskAndMappedColor.pcd");
                pcl::PCDWriter pcd_writer_temp;
                pcd_writer_temp.writeBinary(cloudInWorldWithMaskandMappedColorDir, *cloudInWorldWithMaskandMappedColor);
            }
        }
    }
    else
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
                cout << "All colored cloud saved to: " << cloudInWorldWithRGB << endl;
            }
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

        // TODO Implement logic to find the mask image path for the given timestamp

        if (enableMaskSegmentation)
        {
            std::string maskImagePath = maskImageFolder + std::to_string(timestamp) + ".png";
            if (!imagePath.empty())
            {
                FrameData frame(imagePath, timestamp, pose);
                frame.addSegmentImage(maskImagePath);
                frames.push_back(frame);
            }
        }
        else
        {
            frames.emplace_back(imagePath, timestamp, pose);
        }
    }
}

void PointCloudProcessor::process()
{
    loadPointCloud();
    loadImagesAndOdometry();

    // create the output folder "filtered_pcd/"
    {
        std::string filteredPcdFolderPath = outputPath + "filtered_pcd/";

        std::filesystem::path outputPcdDir(filteredPcdFolderPath);

        if (std::filesystem::exists(outputPcdDir))
        {
            std::filesystem::remove_all(outputPcdDir); // Delete the folder and all its contents
        }

        std::filesystem::create_directories(outputPcdDir); // Create the folder (and any necessary parent directories)
    }

    bool isKeyframe = true;
    // Initialize keyframe identification variables
    FrameData *previousFrame = nullptr;
    const double distThreshold = 0.8; // meter, 1
    const double angThreshold = 30.0; // degree. 25

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
