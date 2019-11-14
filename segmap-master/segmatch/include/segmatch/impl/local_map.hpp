#ifndef SEGMATCH_IMPL_LOCAL_MAP_HPP_
#define SEGMATCH_IMPL_LOCAL_MAP_HPP_

#include "segmatch/local_map.hpp"

#include "laser_slam/benchmarker.hpp"

#include "segmatch/common.hpp"
#include "segmatch/dynamic_voxel_grid.hpp"
#include "segmatch/points_neighbors_providers/kdtree_points_neighbors_provider.hpp"
#include "segmatch/points_neighbors_providers/octree_points_neighbors_provider.hpp"

namespace segmatch {

// Force the compiler to reuse instantiations provided in local_map.cpp
extern template class LocalMap<PclPoint, MapPoint>;

//=================================================================================================
//    LocalMap public methods implementation
//=================================================================================================

template<typename InputPointT, typename ClusteredPointT>
LocalMap<InputPointT, ClusteredPointT>::LocalMap(
    const LocalMapParameters& params, std::unique_ptr<NormalEstimator> normal_estimator)
  : voxel_grid_(params.voxel_size_m, params.min_points_per_voxel)
  , radius_squared_m2_(pow(params.radius_m, 2.0))
  , min_vertical_distance_m_(params.min_vertical_distance_m)
  , max_vertical_distance_m_(params.max_vertical_distance_m)
  , normal_estimator_(std::move(normal_estimator)) {

  // Create the points neighbors provider.
  // 创建邻域提取对象
  if (params.neighbors_provider_type == "KdTree") {
    points_neighbors_provider_ = std::unique_ptr<PointsNeighborsProvider<ClusteredPointT>>(
        new KdTreePointsNeighborsProvider<ClusteredPointT>());
  } else if (params.neighbors_provider_type == "Octree") {
    points_neighbors_provider_ = std::unique_ptr<PointsNeighborsProvider<ClusteredPointT>>(
        new OctreePointsNeighborsProvider<ClusteredPointT>(params.voxel_size_m));
  } else {
    LOG(ERROR) << "Invalid points neighbors provider type specified: "
        << params.neighbors_provider_type;
    throw std::invalid_argument("Invalid points neighbors provider type specified: " +
                                params.neighbors_provider_type);
  }
}

template<typename InputPointT, typename ClusteredPointT>
void LocalMap<InputPointT, ClusteredPointT>::updatePoseAndAddPoints(
    const std::vector<InputCloud>& new_clouds, const laser_slam::Pose& pose) {
  BENCHMARK_BLOCK("SM.UpdateLocalMap");
  
  // 根据新位姿更新局部地图体素，经此处理后，voxel可能会发生变化（减少）
  // 返回：激活体素是否移除标志序列（true为已移除，序列长度为移除之前activate部分长度）
  std::vector<bool> is_point_removed = updatePose(pose);
  // 将点云更新到局部栅格中，经此处理后，voxel再次变化（增加）
  // 返回：新加激活体素的在active_centroids_中的索引
  std::vector<int> created_points_indices = addPointsAndGetCreatedVoxels(new_clouds);
  // 返回activate_voxel_处理前后的索引映射
  std::vector<int> points_mapping = buildPointsMapping(is_point_removed, created_points_indices);

  // Update the points neighbors provider.
  BENCHMARK_START("SM.UpdateLocalMap.UpdatePointsNeighborsProvider");
  // 相当于points_neighbors_provider_.update，更新邻域搜索的源点云
  getPointsNeighborsProvider().update(getFilteredPointsPtr(), {});
  BENCHMARK_STOP("SM.UpdateLocalMap.UpdatePointsNeighborsProvider");

  // If required, update the normals.
  // 该版本代码中不需要 normal_estimator_=nullptr
  if (normal_estimator_ != nullptr) {
    BENCHMARK_BLOCK("SM.UpdateLocalMap.EstimateNormals");
    is_normal_modified_since_last_update_ = normal_estimator_->updateNormals(
        getFilteredPoints(), points_mapping, created_points_indices, getPointsNeighborsProvider());
  } else {
    is_normal_modified_since_last_update_ = std::vector<bool>(getFilteredPoints().size(), false);
  }
}

// 模板的具体类型，见localmap的声明
// local_map维护了一个半径为radius_m（50）的局部地图
// 根据新的位姿，移除在半径范围外的体素，并返回移除信息
template<typename InputPointT, typename ClusteredPointT>
std::vector<bool> LocalMap<InputPointT, ClusteredPointT>::updatePose(const laser_slam::Pose& pose) {
  BENCHMARK_BLOCK("SM.UpdateLocalMap.UpdatePose");

  pcl::PointXYZ position;
  position.x = pose.T_w.getPosition()[0];
  position.y = pose.T_w.getPosition()[1];
  position.z = pose.T_w.getPosition()[2];

  // Remove points according to a cylindrical filter predicate.
  // 根据输入点，通过圆柱滤波删除周围栅格
  // 向removeIf传入函数：在指定半径范围外，返回true
  // is_point_removed是激活体素中移除情况的标志序列
  std::vector<bool> is_point_removed = voxel_grid_.removeIf([&](const ClusteredPointT& p) {
    float distance_xy_squared = pow(p.x - position.x, 2.0) + pow(p.y - position.y, 2.0);
	// 参数中radius_squared_m2_为50的平方
	// min_vertical_distance_m_为-999
	// max_vertical_distance_m_为999
    bool remove = distance_xy_squared > radius_squared_m2_
        || p.z - position.z < min_vertical_distance_m_
        || p.z - position.z > max_vertical_distance_m_;
    // TODO: Once we start supporting multiple segmenters working on the same cloud, we will need
    // one \c segment_ids_ vector per segmenter.
	// 如果基于同一点云，多个分割器工作，每个分割器都要有分割id队列
    if (remove && p.ed_cluster_id != 0u)
      segment_ids_[p.ed_cluster_id] = kInvId;
    if (remove && p.sc_cluster_id != 0u)
      segment_ids_[p.sc_cluster_id] = kInvId;
    return remove;
  });

  return is_point_removed;
}

// 在体素栅格中加入点云，返回新激活的体素索引序列
// 返回：新加入的激活体素的索引
template<typename InputPointT, typename ClusteredPointT>
std::vector<int> LocalMap<InputPointT, ClusteredPointT>::addPointsAndGetCreatedVoxels(
    const std::vector<InputCloud>& new_clouds) {
  BENCHMARK_BLOCK("SM.UpdateLocalMap.AddNewPoints");

  // Reserve space for the new cloud.
  // 统计点云数据量，预备空间
  InputCloud merged_cloud;
  size_t points_count = 0u;
  for (const auto& cloud : new_clouds) points_count += cloud.size();
  merged_cloud.reserve(points_count);

  // Accumulate clouds and insert them in the voxel grid.
  // 将本次处理累计的点云插入到voxel_grid
  for (const auto& cloud : new_clouds) merged_cloud += cloud;
  // 将点云更新到体素栅格中，得到新加入的激活体素的索引
  std::vector<int> created_points_indices = voxel_grid_.insert(merged_cloud);

  // Record local map metrics.
  // 记录局部地图兔粮
  BENCHMARK_RECORD_VALUE("SM.UpdateLocalMap.InsertedPoints", merged_cloud.size());
  BENCHMARK_RECORD_VALUE("SM.UpdateLocalMap.CreatedVoxels", created_points_indices.size());
  BENCHMARK_RECORD_VALUE("SM.UpdateLocalMap.ActiveVoxels", getFilteredPoints().size());
  BENCHMARK_RECORD_VALUE("SM.UpdateLocalMap.InactiveVoxels",
                         voxel_grid_.getInactiveCentroids().size());

  return created_points_indices;
}

// 构建activate_voxel_处理前后的索引映射
template<typename InputPointT, typename ClusteredPointT>
std::vector<int> LocalMap<InputPointT, ClusteredPointT>::buildPointsMapping(
    const std::vector<bool>& is_point_removed, const std::vector<int>& new_points_indices) {
  BENCHMARK_BLOCK("SM.UpdateLocalMap.BuildPointsMapping");

  // Build a mapping from index in the old point cloud to index in the new point cloud.
  // 构建旧点云中的索引到新点云中的索引的映射
  size_t new_point_index = 0u;
  size_t next_inserted_point_index = 0u;
  std::vector<int> mapping(is_point_removed.size());

  for (size_t old_point_index = 0u; old_point_index < is_point_removed.size(); ++old_point_index) {
    if (is_point_removed[old_point_index]) {
      // Mark point as removed.
	  // 标记移除的点
      mapping[old_point_index] = -1;
    } else {
      while (next_inserted_point_index < new_points_indices.size() &&
          new_points_indices[next_inserted_point_index] == new_point_index) {
        // Skip any inserted point, they don't belong to the mapping.
		// 跳过新加入的点，这些点没有映射
        ++new_point_index;
        ++next_inserted_point_index;
      }
      mapping[old_point_index] = new_point_index++;
    }
  }

  // 最终返回的是处理前activate_voxel_到处理后activate_voxel_索引的映射
  // 如果已删除则赋值为-1
  return mapping;
}

template<typename InputPointT, typename ClusteredPointT>
void LocalMap<InputPointT, ClusteredPointT>::transform(
    const kindr::minimal::QuatTransformationTemplate<float>& transformation) {
  BENCHMARK_BLOCK("SM.TransformLocalMap");
  voxel_grid_.transform(transformation);

  if (normal_estimator_ != nullptr) {
    BENCHMARK_BLOCK("SM.TransformLocalMap.TransformNormals");
    normal_estimator_->notifyPointsTransformed(transformation);
  }
}

template<typename InputPointT, typename ClusteredPointT>
void LocalMap<InputPointT, ClusteredPointT>::clear() {
  voxel_grid_.clear();
  if (normal_estimator_ != nullptr)
    normal_estimator_->clear();
}

} // namespace segmatch

#endif // SEGMATCH_IMPL_LOCAL_MAP_HPP_
