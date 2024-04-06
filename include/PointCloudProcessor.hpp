#ifndef POINTCLOUDPROCESSOR_HPP
#define POINTCLOUDPROCESSOR_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>
#include "RGBFrames.hpp"
#include "FrameData.hpp"

class PointCloudProcessor {
public:
    PointCloudProcessor(const std::string& pointCloudPath, const std::string& odometryPath, const std::string& imagesFolder);

    void process();

private:
    std::string pointCloudPath;
    std::string odometryPath;
    std::string imagesFolder;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    std::vector<FrameData> frames;

    void loadPointCloud();
    void loadVisualOdometry();
    void loadImages();
    void applyFOVDetectionAndHiddenPointRemoval();
    void colorizePoints();
    void smoothColors();
    void saveColorizedPointCloud(const std::string& outputPath);
};

#endif // POINTCLOUDPROCESSOR_HPP
