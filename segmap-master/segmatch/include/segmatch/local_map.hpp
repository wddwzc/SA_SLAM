#ifndef SEGMATCH_LOCAL_MAP_HPP_
#define SEGMATCH_LOCAL_MAP_HPP_

#include <laser_slam/common.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/io.h>
#include <pcl/point_cloud.h>

#include "segmatch/common.hpp"
#include "segmatch/dynamic_voxel_grid.hpp"
#include "segmatch/normal_estimators/normal_estimator.hpp"
#include "segmatch/points_neighbors_providers/points_neighbors_provider.hpp"

namespace segmatch {

/// \brief Parameters of the local map.
struct LocalMapParameters {
  /// \brief Size of a voxel in the grid.
  // 体素栅格尺寸
  float voxel_size_m;
  /// \brief Minimum number of points that a voxel must contain in order to be
  /// considered active.
  // 激活体素所需最小点数
  int min_points_per_voxel;
  /// \brief Radius of the local map.
  // 局部地图半径
  float radius_m;
  /// \brief Minimum vertical distance between a point and the robot.
  // 点和机器人间的最小垂直距离
  float min_vertical_distance_m;
  /// \brief Maximum vertical distance between a point and the robot.
  // 点和机器人间的最大垂直距离
  float max_vertical_distance_m;
  /// \brief Type of the method used for querying nearest neighbors information.
  // 用于查询最近邻信息的方法类型
  std::string neighbors_provider_type;
};

/// \brief Manages the local point cloud of a robot. Provides methods for inserting, filtering and
/// segmenting points.
/// \remark The class is \e not thread-safe. Concurrent access to the class results in undefined
/// behavior.
// 管理机器人的局部点云。提供插入、滤波、分割点的方法
// 该类不是线程安全的，不能并发
template<typename InputPointT, typename ClusteredPointT>
class LocalMap {
 public:
  typedef DynamicVoxelGrid<InputPointT, ClusteredPointT> VoxelGrid;
  typedef typename VoxelGrid::InputCloud InputCloud;
  typedef typename VoxelGrid::VoxelCloud ClusteredCloud;

  /// \brief Initializes a new instance of the LocalMap class.
  /// \param params The parameters of the local map.
  /// \param normal_estimator Pointer to an object that can be used for estimating the normals. If
  /// null, normals will not be estimated.
  // 初始化
  // 参数：局部地图参数  法线估计对象的指针
  LocalMap(const LocalMapParameters& params, std::unique_ptr<NormalEstimator> normal_estimator);

  /// \brief Move constructor for the LocalMap class.
  /// \param other The object to be moved in this instance.
  // 拷贝构造函数
  // 参数：要拷贝的对象
  LocalMap(LocalMap&& other)
    : voxel_grid_(std::move(other.voxel_grid_))
    , radius_squared_m2_(other.radius_squared_m2_)
    , min_vertical_distance_m_(other.min_vertical_distance_m_)
    , max_vertical_distance_m_(other.max_vertical_distance_m_)
    , points_neighbors_provider_(std::move(other.points_neighbors_provider_))
    , normal_estimator_(std::move(other.normal_estimator_)) {
  };

  /// \brief Update the pose of the robot and add new points to the local map.
  /// \param new_clouds Vector of point clouds to be added.
  /// \param pose The new pose of the robot.
  // 更新机器人位置，根据新的位置，更新局部体素栅格，在局部地图中添加新体素，移除无效体素
  // 参数：要添加的点云  机器人的新位姿
  void updatePoseAndAddPoints(const std::vector<InputCloud>& new_clouds,
                              const laser_slam::Pose& pose);

  /// \brief Apply a pose transformation to the points contained in the local map.
  /// \remark Multiple transformations are cumulative.
  /// \param transformation The transformation to be applied to the local map.
  // 对局部地图总的点进行位姿变换（多次变换是累计的）
  // 参数：要应用于局部地图的变换
  void transform(const kindr::minimal::QuatTransformationTemplate<float>& transformation);

  /// \brief Clears the local map, removing all the points it contains.
  // 清空局部地图
  void clear();

  /// \brief Gets a filtered view of the points contained in the point cloud.
  /// \remark Modifying the X, Y, Z components of the points in the returned cloud results in
  /// undefined behavior.
  /// \return Reference to the clustered cloud.
  // 获取点云经过滤波的视图（返回值不能修改）
  // 返回：active_centroids_
  ClusteredCloud& getFilteredPoints() const {
    return voxel_grid_.getActiveCentroids();
  }

  /// \brief Gets a filtered view of the points contained in the point cloud.
  /// \remark Modifying the X, Y, Z components of the points in the returned cloud results in
  /// undefined behavior.
  /// \return Pointer to the clustered cloud.
  // 获取点云经过滤波的视图（返回值不能修改）
  // 返回：ClusteredCloud::ConstPtr
  typename ClusteredCloud::ConstPtr getFilteredPointsPtr() const {
    return typename ClusteredCloud::ConstPtr(&voxel_grid_.getActiveCentroids(),
                                             [](ClusteredCloud const* ptr) {});
  }

  /// \brief Gets the normals of the points of the local map.
  /// \remark The returned value is valid only if the local map has been constructed with the
  /// \c estimate_normals option set to true. Otherwise, the normal cloud is empty.
  /// \return Normals of the points of the map.
  // 获取局部地图点的法线
  // 返回值只在局部地图estimate_normals选项为true时有效，否则点云为空
  // 返回：地图点的法线
  const PointNormals& getNormals() const {
    if (normal_estimator_ != nullptr)
      return normal_estimator_->getNormals();
    else
      return empty_normals_cloud_;
  }

  /// \brief Gets an object that can be used for nearest neighbors queries on the points of the
  /// local map.
  /// \returns The points neighbors provider object.
  // 获取一个用于对局部地图进行最近邻搜索的对象
  // 返回：点云最近邻搜索对象
  inline PointsNeighborsProvider<ClusteredPointT>& getPointsNeighborsProvider() {
    return *points_neighbors_provider_;
  }

  /// \brief Gets a reference to the current mapping from clusters to segment IDs. Cluster \c i has
  /// segment ID <tt>getClusterToSegmentIdMapping()[i]</tt>.
  /// \return Reference to the mapping between clusters and segment IDs.
  // 获取聚类到分块ID的映射引用，segment_ids_[i]就是cluster i对应的segment IDs
  // 返回：映射的引用
  std::vector<Id>& getClusterToSegmentIdMapping() {
    return segment_ids_;
  }

  /// \brief Gets the indices of the normals that have been modified since the last update.
  /// \returns Indices of the modified normals.
  // 获取自上次更新以来被修改的法线的索引
  // 返回：被修改的法线索引
  std::vector<bool> getIsNormalModifiedSinceLastUpdate() {
    return is_normal_modified_since_last_update_;
  }

 private:
  std::vector<bool> updatePose(const laser_slam::Pose& pose);
  std::vector<int> addPointsAndGetCreatedVoxels(const std::vector<InputCloud>& new_clouds);
  std::vector<int> buildPointsMapping(const std::vector<bool>& is_point_removed,
                                      const std::vector<int>& new_points_indices);

  VoxelGrid voxel_grid_;

  const float radius_squared_m2_;
  const float min_vertical_distance_m_;
  const float max_vertical_distance_m_;

  std::unique_ptr<PointsNeighborsProvider<ClusteredPointT>> points_neighbors_provider_;
  std::unique_ptr<NormalEstimator> normal_estimator_;
  PointNormals empty_normals_cloud_;

  // Variables needed for working with incremental updates.
  std::vector<Id> segment_ids_;
  std::vector<bool> is_normal_modified_since_last_update_;
}; // class LocalMap

} // namespace segmatch

#endif // SEGMATCH_LOCAL_MAP_HPP_
