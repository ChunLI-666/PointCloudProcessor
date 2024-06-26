#ifndef FRAMEDATA_HPP
#define FRAMEDATA_HPP

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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

class FrameData
{
public:
  Pose pose;
  cv::Mat image;
  std::string imagePath;
  double imageTimestamp; // Add this line

  cv::Mat maskImage;
  std::string maskImagePath;

  FrameData(const std::string &imagePath, double image_timestamp, const Pose &pose);

  void addSegmentImage(const std::string &maskImagePath);
  // addImage(cv::Mat image, double timestamp);
};

#endif // FRAMEDATA_HPP
