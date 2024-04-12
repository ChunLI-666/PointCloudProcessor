#ifndef FRAMEDATA_HPP
#define FRAMEDATA_HPP

#include <opencv2/opencv.hpp>

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

class FrameData
{
public:
  Pose pose;
  cv::Mat image;
  std::string imagePath;
  double imageTimestamp; // Add this line
  FrameData(const std::string &imagePath, double image_timestamp, const Pose &pose);

  // addImage(cv::Mat image, double timestamp);
};

#endif // FRAMEDATA_HPP
