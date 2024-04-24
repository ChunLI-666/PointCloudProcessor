#ifndef RGBFRAMES_HPP
#define RGBFRAMES_HPP

#include <memory>
#include <unordered_map>
#include "FrameData.hpp"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// struct RGBPoint
// {
//     float x, y, z;
//     std::unordered_map<int, std::shared_ptr<FrameData>> frames;
// };

// Define a new point type that inherits from pcl::PointXYZRGB and adds a new float field
struct PointXYZRGBMask : public pcl::PointXYZRGB
{
  float segmentMask;
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // Ensure proper alignment for performance reasons
} EIGEN_ALIGN16;  // Enforces SSE padding for correct memory alignment

// Register the new point type with Point Cloud Library
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBMask,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (std::uint8_t, r, r)
                                  (std::uint8_t, g, g)
                                  (std::uint8_t, b, b)
                                  (float, segmentMask, segmentMask)
)

class RGBFrames
{
public:
    void addFrame(const FrameData &frame);

private:
    std::vector<FrameData> frames;
};

#endif // RGBFRAMES_HPP
