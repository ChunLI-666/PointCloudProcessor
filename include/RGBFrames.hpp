#ifndef RGBFRAMES_HPP
#define RGBFRAMES_HPP

#include <memory>
#include <unordered_map>
#include "FrameData.hpp"

struct RGBPoint {
    float x, y, z;
    std::unordered_map<int, std::shared_ptr<FrameData>> frames;
};

class RGBFrames {
public:
    void addFrame(const std::shared_ptr<FrameData>& frame);
private:
    std::vector<std::shared_ptr<FrameData>> frames;
};

#endif // RGBFRAMES_HPP
