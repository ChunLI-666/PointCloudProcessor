#include <atomic>
#include <thread>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>

// #include <vlcal/calib/visual_camera_calibration.hpp>
#include "FrameData.hpp"

#include <camera/create_camera.hpp>
#include <vlcal/costs/nid_cost.hpp>
// #include <vlcal/common/console_colors.hpp>
// #include <vlcal/common/visual_lidar_data.hpp>
// #include <vlcal/common/points_color_updater.hpp>
// #include <vlcal/common/visual_lidar_visualizer.hpp>
#include <vlcal/calib/visual_camera_calibration.hpp>


namespace vlcal
{

  class VisualLiDARCalibration
  {
  public:
    VisualLiDARCalibration(const std::string &camera_model,
                           const std::vector<double> &K_camera,
                           const std::vector<double> &D_camera,
                          //  const pcl::PointCloud<pcl::PointXYZI>::Ptr &point_cloud,
                           const FrameData &frame)
        : camera_model(camera_model),
          K_camera(K_camera),
          D_camera(D_camera),
          // pcl_cloud_filtered(point_cloud),
          frame(frame)
    {
      // Creare camera projection model
      proj = camera::create_camera(camera_model, K_camera, D_camera);
      T_camera_lidar = Eigen::Isometry3d::Identity();
    }

    void calibrate(const pcl::PointCloud<pcl::PointXYZI>::Ptr &point_cloud_origin, pcl::PointCloud<pcl::PointXYZI>::Ptr &point_cloud_out)
    {
      // std::vector<double> init_values{7, 0.0}; // vector of init T_lidar_camera
      // Init initial values with identity matrix. In our case, the point cloud is already in camera frame
      // init_values = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}; // 7 elements (x, y, z, qx, qy, qz, qw)

      // if (init_values.empty())
      // {
      //   std::cerr << "error: initial guess of T_lidar_camera must be computed before calibration!!" << std::endl;
      //   abort();
      // }

      // assign the initial guess of T_lidar_camera
      Eigen::Isometry3d init_T_lidar_camera = Eigen::Isometry3d::Identity();
      // init_T_lidar_camera.translation() << init_values[0], init_values[1], init_values[2];
      // init_T_lidar_camera.linear() == Eigen::Quaterniond(init_values[6], init_values[3], init_values[4], init_values[5]).normalized().toRotationMatrix();

      const Eigen::Isometry3d init_T_camera_lidar = init_T_lidar_camera.inverse();

      VisualCameraCalibrationParams params;
      // params.disable_z_buffer_culling = vm.count("disable_culling");
      params.nid_bins = 16;
      params.nelder_mead_init_step = 0.001;
      params.nelder_mead_convergence_criteria = 1e-8;

      const std::string registration_type = "nid_bfgs";
      if (registration_type == "nid_bfgs")
      {
        params.registration_type = RegistrationType::NID_BFGS;
      }
      else if (registration_type == "nid_nelder_mead")
      {
        params.registration_type = RegistrationType::NID_NELDER_MEAD;
      }
      else
      {
        std::cerr << "warning: unknown registration type " << registration_type << std::endl;
      }

      VisualCameraCalibration calib(proj, frame, params, point_cloud_origin);  

      std::atomic_bool optimization_terminated = false;
      // Eigen::Isometry3d T_camera_lidar = init_T_camera_lidar;
      T_camera_lidar = init_T_camera_lidar;
      std::thread optimization_thread([&]
                                      {
            T_camera_lidar = calib.calibrate(init_T_camera_lidar);
            optimization_terminated = true; });

      optimization_thread.join();

      // Save the optimized results
      const Eigen::Isometry3d T_lidar_camera = T_camera_lidar.inverse(); //TODO： check if this is correct
      // const Eigen::Vector3d trans(T_lidar_camera.translation());
      // const Eigen::Quaterniond quat(T_lidar_camera.linear());

      // const std::vector<double> T_lidar_camera_values = {trans.x(), trans.y(), trans.z(), quat.x(), quat.y(), quat.z(), quat.w()};

      // Clear the output point cloud
      point_cloud_out->clear();
      
      // Iterate through each point in the original point cloud, copy the intensity and transformed point,
      // and push back transformed point into the point_cloud_out  
      for (const auto& point: point_cloud_origin->points){
        // Transform the point from lidar frame to camera frame with the optimized T_lidar_camera
        Eigen::Vector3d pt_lidar(point.x, point.y, point.z);
        Eigen::Vector3d pt_transformed = T_camera_lidar * pt_lidar;

        pcl::PointXYZI transformed_point;
        transformed_point.x = pt_transformed.x();
        transformed_point.y = pt_transformed.y();
        transformed_point.z = pt_transformed.z();
        transformed_point.intensity = point.intensity;

        point_cloud_out->push_back(transformed_point); 
      }
    }

    Eigen::Isometry3d getOptimizedPose(){
      return T_camera_lidar;      
    }

  private:
    const std::string camera_model;
    const std::vector<double> K_camera;
    const std::vector<double> D_camera;
    const pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud_filtered;
    camera::GenericCameraBase::ConstPtr proj;
    // std::vector<VisualLiDARData::ConstPtr> dataset;
    const FrameData frame;
    Eigen::Isometry3d T_camera_lidar;
  };

} // namespace vlcal

// int main(int argc, char **argv)
// {
//   using namespace boost::program_options;
//   options_description description("calibrate");

//  VisualCameraCalibration // clang-format off
//   description.add_options()
//     ("help", "produce help message")
//     ("data_path", value<std::string>(), "directory that contains preprocessed data")
//     ("first_n_bags", value<int>(), "use only the first N bags (just for evaluation)")
//     ("disable_culling", "disable depth buffer-based hidden points removal")
//     ("nid_bins", value<int>()->default_value(16), "Number of histogram bins for NID")
//     ("registration_type", value<std::string>()->default_value("nid_bfgs"), "nid_bfgs or nid_nelder_mead")
//     ("nelder_mead_init_step", value<double>()->default_value(1e-3), "Nelder-mead initial step size")
//     ("nelder_mead_convergence_criteria", value<double>()->default_value(1e-8), "Nelder-mead convergence criteria")
//     ("auto_quit", "automatically quit after calibration")
//     ("background", "hide viewer and run calibration in background")
//   ;
//   // clang-format on

//   positional_options_description p;
//   p.add("data_path", 1);

//   variables_map vm;
//   store(command_line_parser(argc, argv).options(description).positional(p).run(), vm);
//   notify(vm);

//   if (vm.count("help") || !vm.count("data_path"))
//   {
//     std::cout << description << std::endl;
//     return 0;
//   }

//   const std::string data_path = vm["data_path"].as<std::string>();

//   // params should include:
//   // 1. camera model, camera intrinsics, distortion coefficients
//   // 2. initial guess of T_lidar_camera, in this case is T_lidar_camera = I, since point cloud is in camera frame
//   // 3. Visible point clouds
//   // 4. raw image frame

//   vlcal::VisualLiDARCalibration calib(data_path, vm);
//   calib.calibrate(vm);

//   return 0;
// }