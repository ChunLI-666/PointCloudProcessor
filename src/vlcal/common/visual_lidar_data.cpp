#include <vlcal/common/visual_lidar_data.hpp>

#include <opencv2/opencv.hpp>
#include <vlcal/common/console_colors.hpp>

#include <glk/io/ply_io.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>          // For loading point cloud

namespace vlcal {

VisualLiDARData::VisualLiDARData(const std::string& data_path, const std::string& bag_name) {
  std::cout << "loading " << data_path + "/" + bag_name + ".(png|ply)" << std::endl;

  image = cv::imread(data_path + "/" + bag_name + ".png", 0);
  if (!image.data) {
    std::cerr << vlcal::console::bold_red << "warning: failed to load " << data_path + "/" + bag_name + ".png" << vlcal::console::reset << std::endl;
    abort();
  }

  auto ply = glk::load_ply(data_path + "/" + bag_name + ".ply");
  if (!ply) {
    std::cerr << vlcal::console::bold_red << "warning: failed to load " << data_path + "/" + bag_name + ".ply" << vlcal::console::reset << std::endl;
    abort();
  }

  points = std::make_shared<FrameCPU>(ply->vertices);
  points->add_intensities(ply->intensities);
}

VisualLiDARData::VisualLiDARData(const FrameData& keyframe) {
  // std::cout << "loading " << data_path + "/" + bag_name + ".(png|ply)" << std::endl;

  // image = cv::imread(data_path + "/" + bag_name + ".png", 0);
  // if (!image.data) {
  //   std::cerr << vlcal::console::bold_red << "warning: failed to load " << data_path + "/" + bag_name + ".png" << vlcal::console::reset << std::endl;
  //   abort();
  // }

  // auto ply = glk::load_ply(data_path + "/" + bag_name + ".ply");
  // if (!ply) {
  //   std::cerr << vlcal::console::bold_red << "warning: failed to load " << data_path + "/" + bag_name + ".ply" << vlcal::console::reset << std::endl;
  //   abort();
  // }

  // points = std::make_shared<FrameCPU>(ply->vertices);
  // points->add_intensities(ply->intensities);
  std::cout << "Loading " << keyframe.imagePath << std::endl;

  image = cv::imread(keyframe.imagePath, 0);
  if (!image.data) {
    std::cerr << vlcal::console::bold_red << "warning: failed to load " << keyframe.imagePath << vlcal::console::reset << std::endl;
    abort();
  }

  auto pcd = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(keyframe.culledPCDPath, *pcd) == -1)
  {
    std::cerr << vlcal::console::bold_red << "warning: failed to load " << keyframe.culledPCDPath << vlcal::console::reset << std::endl;
  }
  std::cout << "Loaded point cloud with " << pcd->points.size() << " points." << std::endl;

  points = std::make_shared<FrameCPU>(pcd);

  // Extract intensities from point cloud pcd
  std::vector<float> intensities;
  intensities.reserve(pcd->points.size());

  for (const auto& p : pcd->points) {
      intensities.push_back(p.intensity);
  }

  points->add_intensities(intensities);

}

VisualLiDARData::~VisualLiDARData() {}

}