#ifndef LASER_SLAM_LASER_TRACK_HPP_
#define LASER_SLAM_LASER_TRACK_HPP_

#include <mutex>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <mincurves/DiscreteSE3Curve.hpp>

#include "laser_slam/common.hpp"
#include "laser_slam/parameters.hpp"

namespace laser_slam {

/// \brief The LaserTrack class interfaces point cloud acquisition with Sliding Window Estimator
/// problem setup and optimization.
// 
class LaserTrack {

 public:
  LaserTrack() {};

  /// \brief Constructor.
  explicit LaserTrack(const LaserTrackParams& parameters, unsigned int laser_track_id = 0u);

  ~LaserTrack() {};

  /// \brief Process a new pose measurement in world frame.
  // 处理世界坐标系中一个新位姿的测量
  void processPose(const Pose& pose);

  /// \brief Process a new laser scan in laser frame.
  // 处理激光坐标系中一个新激光扫描
  void processLaserScan(const LaserScan& scan);

  void processPoseAndLaserScan(const Pose& pose, const LaserScan& in_scan,
                               gtsam::NonlinearFactorGraph* newFactors = NULL,
                               gtsam::Values* newValues = NULL,
                               bool* is_prior = NULL);

  // Accessing the laser data
  /// \brief Get the point cloud of the last laser scan.
  // 获取最近一次激光扫描的点云
  void getLastPointCloud(DataPoints* out_point_cloud) const;

  /// \brief Get the point cloud within a time interval.
  // 获取一个时间间隔内的点云
  void getPointCloudOfTimeInterval(const std::pair<Time, Time>& times_ns,
                                   DataPoints* out_point_cloud) const;

  /// \brief Get one local cloud in world frame.
  // 获取世界坐标下的局部点云
  void getLocalCloudInWorldFrame(const Time& timestamp, DataPoints* out_point_cloud) const;

  /// \brief Get all laser scans.
  // 获取所有激光扫描
  const std::vector<LaserScan>& getLaserScans() const;

  /// \brief Get the trajectory.
  // 获取轨迹
  void getTrajectory(Trajectory* trajectory) const;

  /// \brief Get the trajectory based only on odometry data.
  // 获取仅基于历程及数据的轨迹
  void getOdometryTrajectory(Trajectory* out_trajectory) const;

  /// \brief Get the covariance matrices.
  // 获取协方差矩阵
  void getCovariances(std::vector<Covariance>* out_covariances) const;

  /// \brief Get the current estimate.
  // 获取当前位姿估计
  Pose getCurrentPose() const;

  Pose getPreviousPose() const;

  /// \brief Get the first valid time of the trajectory.
  // 获取轨迹的第一个有效时间
  Time getMinTime() const;

  /// \brief Get the last valid time of the trajectory.
  // 获取轨迹的最新有效时间
  Time getMaxTime() const;

  /// \brief Get the timestamps of the laser scans.
  // 获取激光扫描的时间戳
  void getLaserScansTimes(std::vector<Time>* out_times_ns) const;

  /// \brief Append the prior factors to the factor graph.
  // 将先验的因子添加到因子图中
  void appendPriorFactors(const curves::Time& prior_time_ns,
                          gtsam::NonlinearFactorGraph* graph) const;

  /// \brief Append the odometry factors to the factor graph.
  // 将里程计因子添加到因子图中
  void appendOdometryFactors(const curves::Time& optimization_min_time_ns,
                             const curves::Time& optimization_max_time_ns,
                             gtsam::noiseModel::Base::shared_ptr noise_model,
                             gtsam::NonlinearFactorGraph* graph) const;

  /// \brief Append the ICP factors to the factor graph.
  // 将ICP因子添加到因子图中
  void appendICPFactors(const curves::Time& optimization_min_time_ns,
                        const curves::Time& optimization_max_time_ns,
                        gtsam::noiseModel::Base::shared_ptr noise_model,
                        gtsam::NonlinearFactorGraph* graph) const;

  /// \brief Append loop closure factors to the factor graph.
  // 将闭环因子添加到因子图中
  void appendLoopClosureFactors(const curves::Time& optimization_min_time_ns,
                                const curves::Time& optimization_max_time_ns,
                                gtsam::noiseModel::Base::shared_ptr noise_model,
                                gtsam::NonlinearFactorGraph* graph) const;

  /// \brief Initialize GTSAM values from the trajectory.
  // 根据轨迹初始化GTSAM值
  void initializeGTSAMValues(const gtsam::KeySet& keys, gtsam::Values* values) const;

  /// \brief Update the trajectory from GTSAM values.
  // 根据GTSAM值更新轨迹
  void updateFromGTSAMValues(const gtsam::Values& values);

  /// \brief Update the covariance matrices from GTSAM values.
  // 根据GTSAM值更新协方差矩阵
  void updateCovariancesFromGTSAMValues(const gtsam::NonlinearFactorGraph& factor_graph,
                                        const gtsam::Values& values);

  /// \brief Get the number of registered laser scans.
  // 获取注册激光扫描的数量
  size_t getNumScans() {
    std::lock_guard<std::recursive_mutex> lock(full_laser_track_mutex_);
    return laser_scans_.size();
  };

  /// \brief Print the underlying trajectory -- only for debugging.
  // 打印底层轨迹——仅用于调试
  void printTrajectory() {
    std::lock_guard<std::recursive_mutex> lock(full_laser_track_mutex_);
    trajectory_.print("Laser track trajectory");
  };

  // Find nearest pose to a givent time.
  // TODO: Make obsolete?
  // 找一个离规定时间最近的姿势
  Pose findNearestPose(const Time& timestamp_ns) const;

  void buildSubMapAroundTime(const curves::Time& time_ns,
                             const unsigned int sub_maps_radius,
                             DataPoints* submap_out) const;

  gtsam::Expression<SE3> getValueExpression(const curves::Time& time_ns) {
    std::lock_guard<std::recursive_mutex> lock(full_laser_track_mutex_);
    return trajectory_.getValueExpression(time_ns);
  };

  SE3 evaluate(const curves::Time& time_ns) const {
    std::lock_guard<std::recursive_mutex> lock(full_laser_track_mutex_);
    return trajectory_.evaluate(time_ns);
  }

  void getScanMatchingTimes(std::map<laser_slam::Time, double>* scan_matching_times) const {
    CHECK_NOTNULL(scan_matching_times);
    *scan_matching_times = scan_matching_times_;
  }

  void saveTrajectory(const std::string& filename) const {
    trajectory_.saveCurveTimesAndValues(filename);
  }

 private:
  typedef curves::DiscreteSE3Curve CurveType;

  // Make a relative pose measurement factor.
  // 生成相对位姿测量因子
  gtsam::ExpressionFactor<SE3>
  makeRelativeMeasurementFactor(const RelativePose& relative_pose_measurement,
                                gtsam::noiseModel::Base::shared_ptr noise_model,
                                const bool fix_first_node = false) const;

  // Make a pose measurement factor.
  // 生成位姿测量因子
  gtsam::ExpressionFactor<SE3>
  makeMeasurementFactor(const Pose& pose_measurement,
                        gtsam::noiseModel::Base::shared_ptr noise_model) const;

  // Compute rigid ICP transformations according to the selected strategy.
  // 根据所选策略计算刚体ICP转换
  void computeICPTransformations();

  // Compute ICP transformation between the last local scan to a concatenation of the
  // previous scans.
  // 计算最新的局部扫描到所关联的历史扫描的ICO转换
  void localScanToSubMap();

  // Get the pose measurements at a given time.
  // 获取所给时间点的位姿测量
  SE3 getPoseMeasurement(const Time& timestamp_ns) const { return findPose(timestamp_ns).T_w; };

  // Set a pose key.
  // 设置一个位姿的key
  void setPoseKey(const Time& timestamp_ns, const Key& key) { findPose(timestamp_ns)->key = key; };

  // Get a pose key.
  // 获取一个位姿的key
  Key getPoseKey(const Time& timestamp_ns) const { return findPose(timestamp_ns).key; };

  // Find a pose at a given time.
  // 找到所给时间处的一个位姿
  Pose* findPose(const Time& timestamp_ns);

  // Find a pose at a given time.
  // 重载
  Pose findPose(const Time& timestamp_ns) const;

  // Wrapper to extend the trajectory cleanly.
  // TODO(Renaud): Move this to curves.
  // 包装器扩展轨迹
  Key extendTrajectory(const Time& timestamp_ns, const SE3& value);

  std::vector<LaserScan>::const_iterator getIteratorToScanAtTime(
      const curves::Time& time_ns) const;

  unsigned int laser_track_id_;

  // TODO move pose_measurements_ to the Trajectory type.
  // Pose measurements in world frame.
  // 世界坐标系下的位姿测量
  PoseVector pose_measurements_;

  // Odometry measurements in laser frame.
  // Obtained from combining and interpolating the pose measurements.
  // 激光坐标系下的里程计测量
  // 由位姿测量值的组合和插值得到。（帧间变化量）
  RelativePoseVector odometry_measurements_;

  // Rigid transformations obtained from ICP between LaserScans.
  // 激光帧间ICP得到的刚体变换
  RelativePoseVector icp_transformations_;

  RelativePoseVector loop_closures_;

  // Laser scans in laser frame.
  // 激光坐标系下的激光扫描
  std::vector<LaserScan> laser_scans_;

  // Underlying trajectory.
  // 
  CurveType trajectory_;

  // TODO replace by standard mutex?
  mutable std::recursive_mutex full_laser_track_mutex_;

  // Covariance matrices.
  // 协方差矩阵
  std::vector<Covariance> covariances_;

  // ICP algorithm object.
  // ICP算法对象
  PointMatcher::ICP icp_;

  // ICP input filters.
  // ICP输入滤波器
  PointMatcher::DataPointsFilters input_filters_;

  // Libpointmatcher rigid transformation.
  // Libpointmatcher刚体变换
  PointMatcher::Transformation* rigid_transformation_;

  // Noise models.
  // 噪声模型
  gtsam::noiseModel::Base::shared_ptr prior_noise_model_;
  gtsam::noiseModel::Base::shared_ptr odometry_noise_model_;
  gtsam::noiseModel::Base::shared_ptr icp_noise_model_;

  std::map<laser_slam::Time, double> scan_matching_times_;

  // Parameters.
  LaserTrackParams params_;

  static constexpr double kDistanceBetweenPriorPoses_m = 100.0;
};

}  // namespace laser_slam

#endif /* LASER_SLAM_LASER_TRACK_HPP_ */
