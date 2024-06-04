#include <vlcal/calib/visual_camera_calibration.hpp>

#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>
#include <ceres/autodiff_first_order_function.h>

#include <sophus/se3.hpp>
#include <sophus/ceres_manifold.hpp>

#include <vlcal/costs/nid_cost.hpp>
#include <vlcal/calib/view_culling.hpp>
// #include <vlcal/calib/cost_calculator_nid.hpp>

#include "FrameData.hpp"

namespace vlcal
{

  VisualCameraCalibration::VisualCameraCalibration(
      const camera::GenericCameraBase::ConstPtr &proj,
      // const std::vector<VisualLiDARData::ConstPtr>& dataset,
      const FrameData &frame,
      const VisualCameraCalibrationParams &params,
      const pcl::PointCloud<pcl::PointXYZI>::Ptr &point_cloud)
      : params(params),
        proj(proj),
        frame(frame),
        point_cloud(point_cloud)
  {
  }

  Eigen::Isometry3d VisualCameraCalibration::calibrate(const Eigen::Isometry3d &init_T_camera_lidar)
  {
    Eigen::Isometry3d T_camera_lidar = init_T_camera_lidar;
        // Outer loop
    for (int i = 0; i < params.max_outer_iterations; i++)
    {
      Eigen::Isometry3d new_T_camera_lidar;
      switch (params.registration_type)
      {
      case RegistrationType::NID_BFGS:
        new_T_camera_lidar = estimate_pose_bfgs(T_camera_lidar);
        break;

      case RegistrationType::NID_NELDER_MEAD:
        new_T_camera_lidar = estimate_pose_nelder_mead(T_camera_lidar);
        break;
      }

      const Eigen::Isometry3d delta = new_T_camera_lidar.inverse() * T_camera_lidar;
      T_camera_lidar = new_T_camera_lidar;

      const double delta_t = delta.translation().norm();
      const double delta_r = Eigen::AngleAxisd(delta.linear()).angle();
      const bool converged = delta_t < params.delta_trans_thresh && delta_r < params.delta_rot_thresh;

      std::cout << boost::format("delta_t: %.3f [m] delta_r: %.3f [rad]") % delta_t % delta_r << std::endl;
      std::cout << (converged ? "Outer loop converged" : "Outer loop not converged, Re-run inner optimization with the new viewpoint") << std::endl;

      if (converged)
      {
        break;
      }
    }

    return T_camera_lidar;
  }

  struct MultiNIDCost
  {
  public:
    MultiNIDCost(const Sophus::SE3d &init_T_camera_lidar) : init_T_camera_lidar(init_T_camera_lidar) {}

    void add(const std::shared_ptr<NIDCost> &cost) { costs.emplace_back(cost); }

    template <typename T>
    bool operator()(const T *params, T *residual) const
    {
      std::vector<double> values(Sophus::SE3d::num_parameters);
      std::transform(params, params + Sophus::SE3d::num_parameters, values.begin(), [](const auto &x)
                     { return get_real(x); });
      const Eigen::Map<const Sophus::SE3d> T_camera_lidar(values.data());
      const Sophus::SE3d delta = init_T_camera_lidar.inverse() * T_camera_lidar;

      if (delta.translation().norm() > 0.2 || Eigen::AngleAxisd(delta.rotationMatrix()).angle() > 2.0 * M_PI / 180.0)
      {
        return false;
      }

      std::vector<bool> results(costs.size());
      std::vector<T> residuals(costs.size());

#pragma omp parallel for
      for (int i = 0; i < costs.size(); i++)
      {
        results[i] = (*costs[i])(params, &residuals[i]);
      }

      for (int i = 1; i < costs.size(); i++)
      {
        residuals[0] += residuals[i];
      }

      *residual = residuals[0];

      return std::count(results.begin(), results.end(), false) == 0;
    }

  private:
    Sophus::SE3d init_T_camera_lidar;
    std::vector<std::shared_ptr<NIDCost>> costs;
  };

  struct IterationCallbackWrapper : public ceres::IterationCallback {
  public:
    IterationCallbackWrapper(const std::function<ceres::CallbackReturnType(const ceres::IterationSummary&)>& callback) : callback(callback) {}

    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) { return callback(summary); }

  private:
    std::function<ceres::CallbackReturnType(const ceres::IterationSummary&)> callback;
  };
  
  Eigen::Isometry3d VisualCameraCalibration::estimate_pose_bfgs(const Eigen::Isometry3d &init_T_camera_lidar)
  {

    // Hidden point removal
    ViewCullingParams view_culling_params;
    view_culling_params.enable_depth_buffer_culling = !params.disable_z_buffer_culling;
    // ViewCulling view_culling(proj, {dataset.front()->image.cols, dataset.front()->image.rows}, view_culling_params);
    ViewCulling view_culling(proj, {4096, 3000}, view_culling_params); // TODO: hardcode

    Sophus::SE3d T_camera_lidar(init_T_camera_lidar.matrix()); // Init extrinsics

    std::vector<std::shared_ptr<NIDCost>> nid_costs; // create nid costs for each frame

    // for (int i = 0; i < dataset.size(); i++) {  // for each frames, first remove hidden points and then create NID cost
    //   // Remove hidden points
    //   // auto culled_points = view_culling.cull(dataset[i]->points, init_T_camera_lidar); // TODO: replace with pcd points
    //   pcl::PointCloud<pcl::PointXYZI>::Ptr culled_points = view_culling.cull(point_cloud, init_T_camera_lidar);

    //   // create normalized image in CV_64FC1 format from original image for NID cost
    //   cv::Mat normalized_image;
    //   // dataset[i]->image.convertTo(normalized_image, CV_64FC1, 1.0 / 255.0); // TODO: replace with raw image frame
    //   cv::Mat orig_image = cv::imread(frame.imagePath);

    //   orig_image.convertTo(normalized_image, CV_64FC1, 1.0 / 255.0);

    //   // create NIDCost object with the normalized image, culled points and projection matrix
    //   std::shared_ptr<NIDCost> nid_cost(new NIDCost(proj, normalized_image, culled_points, params.nid_bins));
    //   // add the NIDCost object to the vector
    //   nid_costs.emplace_back(nid_cost);
    // }

    // TODO: replace dataset.size with colorized deque size
    // Remove hidden points
    // auto culled_points = view_culling.cull(dataset[i]->points, init_T_camera_lidar); // TODO: replace with pcd points
    pcl::PointCloud<pcl::PointXYZI>::Ptr culled_points = view_culling.cull(point_cloud, init_T_camera_lidar);

    // create normalized image in CV_64FC1 format from original image for NID cost
    cv::Mat normalized_image;
    // dataset[i]->image.convertTo(normalized_image, CV_64FC1, 1.0 / 255.0); // TODO: replace with raw image frame
    cv::Mat orig_image = cv::imread(frame.imagePath);

    orig_image.convertTo(normalized_image, CV_64FC1, 1.0 / 255.0);

    // create NIDCost object with the normalized image, culled points and projection matrix
    std::shared_ptr<NIDCost> nid_cost(new NIDCost(proj, normalized_image, culled_points, params.nid_bins));
    // add the NIDCost object to the vector
    nid_costs.emplace_back(nid_cost);

    auto sum_nid = new MultiNIDCost(T_camera_lidar);
    for (const auto &nid_cost : nid_costs)
    {
      sum_nid->add(nid_cost);
    }

    auto cost = new ceres::AutoDiffFirstOrderFunction<MultiNIDCost, Sophus::SE3d::num_parameters>(sum_nid);
    ceres::GradientProblem problem(cost, new Sophus::Manifold<Sophus::SE3>());

    ceres::GradientProblemSolver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.update_state_every_iteration = true;
    options.line_search_direction_type = ceres::BFGS;

    options.callbacks.emplace_back(new IterationCallbackWrapper([&](const ceres::IterationSummary &summary)
                                                                {
    params.callback(Eigen::Isometry3d(T_camera_lidar.matrix()));
    return ceres::CallbackReturnType::SOLVER_CONTINUE; }));

    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(options, problem, T_camera_lidar.data(), &summary);

    // std::stringstream sst;
    // sst << boost::format("Inner optimization (BFGS) terminated after %d iterations") % summary.iterations.size() << std::endl;
    // sst << boost::format("Final cost: %.3f") % summary.final_cost << std::endl;
    // sst << "--- T_camera_lidar ---" << std::endl << T_camera_lidar.matrix();
    // // guik::LightViewer::instance()->append_text(sst.str());
    std::cout << boost::format("Inner optimization (BFGS) terminated after %d iterations") % summary.iterations.size() << std::endl;
    std::cout << boost::format("Final cost: %.3f") % summary.final_cost << std::endl;
    std::cout << "--- T_camera_lidar ---" << std::endl
              << T_camera_lidar.matrix();

    return Eigen::Isometry3d(T_camera_lidar.matrix());
  }
}
