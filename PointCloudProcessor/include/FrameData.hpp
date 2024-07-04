#ifndef FRAMEDATA_HPP
#define FRAMEDATA_HPP

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>

struct Pose
{
  double x, y, z, qw, qx, qy, qz;
};

struct Pose6D
{
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

// Define a new point type that inherits from pcl::PointXYZRGB and adds a new float field
struct PointXYZRGBMask : public pcl::PointXYZRGB
{
  float segmentMask;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Ensure proper alignment for performance reasons
} EIGEN_ALIGN16;                  // Enforces SSE padding for correct memory alignment

// Register the new point type with Point Cloud Library
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBMask,
                                  (float, x, x)(float, y, y)(float, z, z)(std::uint8_t, r, r)(std::uint8_t, g, g)(std::uint8_t, b, b)(float, segmentMask, segmentMask))

// struct RGBCloud {
//     pcl::PointCloud<pcl::PointXYZI>::Ptr cloudNoColor;
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudWithSmoothedColor;
//     std::vector<std::vector<float>> viewColors; // 每个view的RGB颜色
//     std::vector<float> orientationScores; // 每个view的方向分数
//     std::vector<float> distanceScores; // 每个view的距离分数
//     std::vector<float> finalScores; // 每个view的最终分数
// };

class FrameData
{
public:
  using Ptr = std::shared_ptr<FrameData>;
  using ConstPtr = std::shared_ptr<const FrameData>;

  FrameData() {};
  // FrameData(const std::string &imagePath, double image_timestamp, const Pose &pose);
  FrameData(const std::string &imagePath, double timestamp, const Pose &pose)
      : imagePath(imagePath), imageTimestamp(timestamp), pose(pose) {
      visiblePoints.reset(new pcl::PointCloud<pcl::PointXYZI>());
  }
  ~FrameData();

public:
  Pose pose;
  cv::Mat image; //raw image
  std::string imagePath;
  double imageTimestamp;

  cv::Mat segmentImage;
  std::string segmentImagePath;
  
  cv::Mat maskImage;
  std::string maskImagePath;

  pcl::PointCloud<pcl::PointXYZI>::Ptr ptsInCamFrame;
  std::string culledPCDPath; // path to the culled pcd data, with no color, in camera coords

  pcl::PointCloud<pcl::PointXYZI>::Ptr visiblePoints;

  Eigen::Isometry3d initTLidarCamera;
  
  void addSegmentImage(const std::string &maskImagePath);
  // addImage(cv::Mat image, double timestamp);
  void addPts(const pcl::PointCloud<pcl::PointXYZI>::Ptr &points);
  void addManualOptimizedPose(Eigen::Isometry3d T_camera_lidar_optimized);
};

#endif // FRAMEDATA_HPP