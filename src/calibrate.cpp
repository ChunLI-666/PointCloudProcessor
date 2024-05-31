#include <atomic>
#include <thread>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>

#include <vlcal/calib/visual_camera_calibration.hpp>

namespace vlcal
{

  class VisualLiDARCalibration
  {
  public:
    VisualLiDARCalibration(const std::string &camera_model,
                           const std::vector<double> &K_camera,
                           const std::vector<double> &D_camera,
                           const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &point_cloud)
        : camera_model(camera_model),
          K_camera(K_camera),
          D_camera(D_camera)
    {

      // TODO: add camera model, intrinsics, distortion_coeffs
      // const std::string camera_model = "";
      // const std::vector<double> intrinsics = ;
      // const std::vector<double> distortion_coeffs = ;
      proj = camera::create_camera(camera_model, intrinsics, distortion_coeffs);
    }

    void calibrate()
    {
      std::vector<double> init_values; // vector of init T_lidar_camera

      if (init_values.empty())
      {
        std::cerr << "error: initial guess of T_lidar_camera must be computed before calibration!!" << std::endl;
        abort();
      }

      // assign the initial guess of T_lidar_camera
      Eigen::Isometry3d init_T_lidar_camera = Eigen::Isometry3d::Identity();
      init_T_lidar_camera.translation() << init_values[0], init_values[1], init_values[2];
      init_T_lidar_camera.linear() == Eigen::Quaterniond(init_values[6], init_values[3], init_values[4], init_values[5]).normalized().toRotationMatrix();

      const Eigen::Isometry3d init_T_camera_lidar = init_T_lidar_camera.inverse();

      // TODO: involve VisualCameraCalibrationParams.hpp
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

      VisualCameraCalibration calib(proj, dataset, params);

      std::atomic_bool optimization_terminated = false;
      Eigen::Isometry3d T_camera_lidar = init_T_camera_lidar;
      std::thread optimization_thread([&]
                                      {
            T_camera_lidar = calib.calibrate(init_T_camera_lidar);
            optimization_terminated = true; });

      optimization_thread.join();

      // Save the optimized results
      const Eigen::Isometry3d T_lidar_camera = T_camera_lidar.inverse();
      const Eigen::Vector3d trans(T_lidar_camera.translation());
      const Eigen::Quaterniond quat(T_lidar_camera.linear());

      const std::vector<double> T_lidar_camera_values = {trans.x(), trans.y(), trans.z(), quat.x(), quat.y(), quat.z(), quat.w()};

      // TODO： Add api to output the optimized T into the colorization pipeline
    }

  private:
    const std::string camera_model;
    const std::vector<double> K_camera;
    const std::vector<double> D_camera;
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_filtered;
    camera::GenericCameraBase::ConstPtr proj;
    std::vector<VisualLiDARData::ConstPtr> dataset;
  };

} // namespace vlcal

int main(int argc, char **argv)
{
  using namespace boost::program_options;
  options_description description("calibrate");

  // clang-format off
  description.add_options()
    ("help", "produce help message")
    ("data_path", value<std::string>(), "directory that contains preprocessed data")
    ("first_n_bags", value<int>(), "use only the first N bags (just for evaluation)")
    ("disable_culling", "disable depth buffer-based hidden points removal")
    ("nid_bins", value<int>()->default_value(16), "Number of histogram bins for NID")
    ("registration_type", value<std::string>()->default_value("nid_bfgs"), "nid_bfgs or nid_nelder_mead")
    ("nelder_mead_init_step", value<double>()->default_value(1e-3), "Nelder-mead initial step size")
    ("nelder_mead_convergence_criteria", value<double>()->default_value(1e-8), "Nelder-mead convergence criteria")
    ("auto_quit", "automatically quit after calibration")
    ("background", "hide viewer and run calibration in background")
  ;
  // clang-format on

  positional_options_description p;
  p.add("data_path", 1);

  variables_map vm;
  store(command_line_parser(argc, argv).options(description).positional(p).run(), vm);
  notify(vm);

  if (vm.count("help") || !vm.count("data_path"))
  {
    std::cout << description << std::endl;
    return 0;
  }

  const std::string data_path = vm["data_path"].as<std::string>();

  // params should include:
  // 1. camera model, camera intrinsics, distortion coefficients
  // 2. initial guess of T_lidar_camera, in this case is T_lidar_camera = I, since point cloud is in camera frame
  // 3. Visible point clouds
  // 4. raw image frame

  vlcal::VisualLiDARCalibration calib(data_path, vm);
  calib.calibrate(vm);

  return 0;
}