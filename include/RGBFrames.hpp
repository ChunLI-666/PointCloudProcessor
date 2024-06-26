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

class RGBFrames
{
public:
    void addFrame(const FrameData &frame);

private:
    std::vector<FrameData> frames;
};

#endif // RGBFRAMES_HPP
