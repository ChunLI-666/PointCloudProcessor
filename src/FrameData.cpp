#include "FrameData.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

FrameData::FrameData(const std::string& imagePath, double image_timestamp, const Pose& pose)
    : pose(pose), image_timestamp(image_timestamp) {
    
    this->imagePath = imagePath;
    this->image_timestamp = imageTimestamp;
    this->pose = pose;

//    image = cv::imread(imagePath, cv::IMREAD_COLOR);
//    if (image.empty()) {
//        std::cerr << "Error loading image from path: " << imagePath << std::endl;
//        throw std::runtime_error("Failed to load image.");
//    }
}
