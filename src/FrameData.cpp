#include "FrameData.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

FrameData::FrameData(const std::string &imagePath, double image_timestamp, const Pose &pose)
    : pose(pose), imageTimestamp(image_timestamp)
{

    this->imagePath = imagePath;
    this->imageTimestamp = image_timestamp;
    this->pose = pose;

    //    image = cv::imread(imagePath, cv::IMREAD_COLOR);
    //    if (image.empty()) {
    //        std::cerr << "Error loading image from path: " << imagePath << std::endl;
    //        throw std::runtime_error("Failed to load image.");
    //    }
}

void FrameData::addSegmentImage(const std::string &maskImagePath)
{
    this->maskImagePath = maskImagePath;
    // this->maskImage = maskImage;
}

void FrameData::addPts(const pcl::PointCloud<pcl::PointXYZI>::Ptr &points){
    this->ptsInCamFrame = points;
}

FrameData::~FrameData(){

}