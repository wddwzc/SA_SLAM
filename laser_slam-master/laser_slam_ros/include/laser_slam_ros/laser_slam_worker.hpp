#ifndef LASER_SLAM_ROS_LASER_SLAM_WORKER_HPP_
#define LASER_SLAM_ROS_LASER_SLAM_WORKER_HPP_

#include <mutex>

#include <laser_slam/common.hpp>
#include <laser_slam/incremental_estimator.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <tf/transform_listener.h>

#include "laser_slam_ros/GetLaserTrackSrv.h"
#include "laser_slam_ros/common.hpp"

namespace laser_slam_ros {


// 结合Incrementalestimator处理激光点云，得到位姿
class LaserSlamWorker {

 public:
  LaserSlamWorker();
  ~LaserSlamWorker();

  void init(ros::NodeHandle& nh, const LaserSlamWorkerParams& params,
            std::shared_ptr<laser_slam::IncrementalEstimator> incremental_estimator,
            unsigned int worker_id = 0u);

  /// \brief Register the local scans to the sliding window estimator.
  // 将局部扫描登记到滑动窗口估计器
  void scanCallback(const sensor_msgs::PointCloud2& cloud_msg_in);

  /// \brief Publish the robot trajectory (as path) in ROS.
  // 把机器人轨迹发布为ros中的path（用于rviz显示）
  void publishTrajectory(const laser_slam::Trajectory& trajectory,
                         const ros::Publisher& publisher) const;

  /// \brief Publish the map.
  // 发布地图
  void publishMap();

  /// \brief Publish the estimated trajectory and the odometry only based trajectory.
  // 发布估计的轨迹和仅基于里程计的轨迹
  void publishTrajectories();

  void getLocalMapFiltered(laser_slam_ros::PointCloud* local_map_filtered);

  // Get a filtered map and apply map separation if desired.
  // 获取经过筛选的映射，如果需要对地图进行分割
  void getFilteredMap(laser_slam_ros::PointCloud* filtered_map);

  // Get a vector containing the optimized point clouds recorded since
  // the last call to this method. This call clears the point cloud queue.
  // 获取包含自上次调用此方法以来记录的优化点云的序列， 此调用清除点云队列。
  std::vector<laser_slam_ros::PointCloud> getQueuedPoints();

  void clearLocalMap();

  tf::StampedTransform getWorldToOdom();

  void getTrajectory(laser_slam::Trajectory* out_trajectory) const;

  void getOdometryTrajectory(laser_slam::Trajectory* out_trajectory) const;

  void updateLocalMap(const laser_slam::SE3& last_pose_before_update,
                      const laser_slam::Time last_pose_before_update_timestamp_ns);

  /// \brief Computes the transform between a start pose and the pose evaluated at the specified
  /// end timestamp.
  /// \param start_pose The starting pose.
  /// \param end_pose Timestamp of the end pose.
  /// \return Transform between the two poses.
  // 计算起始位姿与在指定的结束时间戳处评估的位姿间的变换
  // 参数：起始位姿  结束位姿的时间戳
  laser_slam::SE3 getTransformBetweenPoses(const laser_slam::SE3& start_pose,
                                           const laser_slam::Time end_pose_timestamp_ns) const;

  void setLockScanCallback(bool new_state);

  void exportTrajectories() const;

  void exportTrajectoryHead(laser_slam::Time head_duration_ns, const std::string& filename) const;

  bool exportTrajectoryServiceCall(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

 private:
  // Convert a tf::StampedTransform to a laser_slam::Pose.
  laser_slam::Pose tfTransformToPose(const tf::StampedTransform& tf_transform);
  // TODO: common.hpp?
  laser_slam::SE3 geometryMsgTransformToSE3(const geometry_msgs::Transform& transform);
  geometry_msgs::Transform SE3ToGeometryMsgTransform(const laser_slam::SE3& transform);

  // Standardize the time so that the trajectory starts at time 0.
  // 标准化时间，使轨迹从时间0开始
  laser_slam::Time rosTimeToCurveTime(const laser_slam::Time& timestamp_ns);

  // Convert time from trajectory base back to ROS base.
  // 将时间从基于轨迹的转换回基于ROS的
  laser_slam::Time curveTimeToRosTime(const laser_slam::Time& timestamp_ns) const;

  // TODO(renaud) : using ros::Time(0) means "use the latest available transform". Might solve your problem in relocalizer?
  // 默认值表示 使用最近的可用的转换
  bool getTransform(const std::string& first_frame,
                    const std::string& second_frame,
                    tf::StampedTransform* transform_ptr,
                    ros::Time transform_time = ros::Time(0));

  bool getLaserTracksServiceCall(laser_slam_ros::GetLaserTrackSrv::Request& request,
                                 laser_slam_ros::GetLaserTrackSrv::Response& response);

 private:
  LaserSlamWorkerParams params_;

  unsigned int worker_id_;

  // TODO make laser_track mutex safe (when loop closures are added).
  std::shared_ptr<laser_slam::LaserTrack> laser_track_;

  // TODO use standard mutex?
  mutable std::recursive_mutex world_to_odom_mutex_;
  mutable std::recursive_mutex local_map_filtered_mutex_;
  mutable std::recursive_mutex local_map_mutex_;

  mutable std::recursive_mutex scan_callback_mutex_;
  bool lock_scan_callback_ = false;

  // Subscribers.
  ros::Subscriber scan_sub_;

  // Publishers.
  ros::Publisher trajectory_pub_;
  ros::Publisher local_map_pub_;
  //  ros::Publisher odometry_trajectory_pub_;
  //  ros::Publisher point_cloud_pub_;
  //  ros::Publisher distant_map_pub_;
  //  ros::Publisher new_fixed_cloud_pub_;

  // Services.
  ros::ServiceServer get_laser_track_srv_;
  ros::ServiceServer export_trajectory_srv_;

  tf::TransformListener tf_listener_;

  // Pointer to the incremental estimator.
  std::shared_ptr<laser_slam::IncrementalEstimator> incremental_estimator_;

  // Contains the map which is estimated by the sliding window.
  // TODO(mattia): switch from local_map_ to local_map_queue_
  // 包含通过滑动窗口估计出的地图
  laser_slam_ros::PointCloud local_map_;
  std::vector<laser_slam_ros::PointCloud> local_map_queue_;

  laser_slam_ros::PointCloud local_map_filtered_;

  // Contains the map which is distant from sensor and assumed to be fixed.
  // If the robot revisits the same environment, the distant_map_and local_map_ will be one
  // above each other, each with same density.
  // 包含远离传感器并假定是固定的地图
  // 如果机器人重新访问相同的环境，则distant_map_和local_map_将彼此叠置，且密度相同
  laser_slam_ros::PointCloud distant_map_;

  // Timestamp to be subtracted to each measurement time so that the trajectory starts at time 0.
  // 在每次测量时间中减去的时间戳，使轨迹从时间0开始
  laser_slam::Time base_time_ns_ = 0;

  // Indicates whether the base time was set.
  // 显示基准时间是否设置
  bool base_time_set_ = false;

	// 最新位姿
  laser_slam::SE3 last_pose_;
  // 显示最新位姿是否设置
  bool last_pose_set_ = false;

  laser_slam::Pose last_pose_sent_to_laser_track_;

  pcl::VoxelGrid<laser_slam_ros::PclPoint> voxel_filter_;

  tf::StampedTransform world_to_odom_;

  static constexpr double kTimeout_s = 0.2;
  static constexpr unsigned int kScanSubscriberMessageQueueSize = 1u;
  static constexpr unsigned int kPublisherQueueSize = 50u;
}; // LaserSlamWorker

} // namespace laser_slam_ros

#endif /* LASER_SLAM_ROS_LASER_SLAM_WORKER_HPP_ */
