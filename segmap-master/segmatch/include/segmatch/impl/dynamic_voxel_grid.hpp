#ifndef SEGMATCH_IMPL_DYNAMIC_VOXEL_GRID_HPP_
#define SEGMATCH_IMPL_DYNAMIC_VOXEL_GRID_HPP_

#include "segmatch/dynamic_voxel_grid.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include <glog/logging.h>
#include <laser_slam/benchmarker.hpp>
#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

#include "segmatch/common.hpp"

namespace segmatch {

// Force the compiler to reuse instantiations provided in dynamic_voxel_grid.cpp
extern template class DynamicVoxelGrid<PclPoint, MapPoint>;

//=================================================================================================
//    DynamicVoxelGrid public methods implementation
//=================================================================================================

// 将new_cloud加入到体素序列voxels_中，同时更新active_centroids_和inactive_centroids_
// 返回：新加入的激活体素在active_centroids_的索引
template<_DVG_TEMPLATE_DECL_>
std::vector<int> DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::insert(const InputCloud& new_cloud) {
  std::vector<int> created_voxel_indices;
  if (new_cloud.empty()) return created_voxel_indices;
  created_voxel_indices.reserve(new_cloud.size());
  // 输入点云，得到升序的索引点云列表
  IndexedPoints_ new_points = indexAndSortPoints_(new_cloud);

  // Create containers and reserve space to prevent reallocation
  std::vector<Voxel_> new_voxels;
  std::unique_ptr<VoxelCloud> new_active_centroids(new VoxelCloud());
  std::unique_ptr<VoxelCloud> new_inactive_centroids(new VoxelCloud());
  new_voxels.reserve(voxels_.size() + new_cloud.size());
  new_active_centroids->reserve(active_centroids_->size() + new_cloud.size());
  new_inactive_centroids->reserve(inactive_centroids_->size() + new_cloud.size());

  // Setup iterators
  auto p_it = new_points.begin();
  auto v_it = voxels_.begin();
  const auto p_end = new_points.end();
  const auto v_end = voxels_.end();

  // Merge points updating the affected voxels.
  // 合并点，更新受影响的体素
  while (!(p_it == p_end && v_it == v_end)) {
    VoxelData_ voxel_data = { nullptr, p_it, p_it};
    IndexT voxel_index;

    // Use the next voxel if it has the upcoming index.
	// 判断：新的点云序列遍历结束 || 旧体素索引小于等于新点索引（升序的）
	// 该语句就是判断是否用已有体素赋值给voxel_data
	// >：执行else分支，加入新点
	// <：只由当前点信息，新加点为空
	// =：二者都有
    if ((p_it == p_end) || (v_it != v_end && v_it->index <= p_it->voxel_index)) {
      voxel_index = v_it->index;
      voxel_data.old_voxel = &(*v_it);
      ++v_it;
    } else {
      voxel_index = p_it->voxel_index;
    }

    // Gather all the points that belong to the current voxel
	// 收集所有属于当前体素的点（索引相同的）
	// 此处如果两个索引不等，就直接跳过了，所以points_begin=points_end
	// 该语句根据voxel_index获取同样索引的新加点
    while (p_it != p_end && p_it->voxel_index == voxel_index) {
      ++p_it;
    }
    voxel_data.points_end = p_it;

    // Create the voxel
	// 根据当前新体素（可能已存在），创建并更新体素序列
	// 更新质心，返回值为：是否为新加入的激活体素
    if (createVoxel_(voxel_index, voxel_data, new_voxels,
                     *new_active_centroids, *new_inactive_centroids)) {
	  // 记录新加入的激活体素在active_centroids_中的索引
      created_voxel_indices.push_back(new_active_centroids->size()-1);
    }
  }

  // Done! Save the new voxels and return the indices of the triggered
  // voxels.
  voxels_= std::move(new_voxels);
  active_centroids_ = std::move(new_active_centroids);
  inactive_centroids_ = std::move(new_inactive_centroids);
  return created_voxel_indices;
}

// 获取点对应的索引（64位索引，包含xyz）
template<_DVG_TEMPLATE_DECL_>
template<typename PointXYZ_>
inline IndexT DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::getIndexOf(const PointXYZ_& point) const {
  static_assert(pcl::traits::has_xyz<PointXYZ_>::value,
                "PointXYZ_ must be a structure containing XYZ coordinates");
  // TODO: One could pack indexing transformation, offsetting and scaling in a single
  // transformation. min_corner and max_corner would need to be transformed as well in order
  // to allow checks. Since it would decrease readability significantly, this should be done only
  // if optimization is really necessary.
  // 可以在单个转换中打包索引转换，偏移和缩放。
  // min_corner和max_corner也需要进行转换才能进行检查。
  // 由于它会显着降低可读性，因此应该只有在真正需要优化时才这样做。

  // Transform the point back to the grid frame for hashing.
  // 点转回栅格坐标
  Eigen::Vector3f transformed_coords = indexing_transformation_.transform(point.getVector3fMap());

  // Ensure that the transformed point lies inside the grid.
  // 确保转换后的点在栅格内
  CHECK(min_corner_(0) <= transformed_coords.x() && transformed_coords.x() < max_corner_(0));
  CHECK(min_corner_(1) <= transformed_coords.y() && transformed_coords.y() < max_corner_(1));
  CHECK(min_corner_(2) <= transformed_coords.z() && transformed_coords.z() < max_corner_(2));

  // Compute voxel index of the point.
  // 计算点的体素索引  (坐标+原点偏移)*坐标转化比例10
  // 索引按位存放在uint64变量中，所以xyz各轴的索引范围为2^20-1=1048576-1
  Eigen::Vector3f grid_coords = (transformed_coords + indexing_offset_) * world_to_grid_;
  return static_cast<IndexT>(grid_coords[0])
      + (static_cast<IndexT>(grid_coords[1]) << bits_x)
      + (static_cast<IndexT>(grid_coords[2]) << (bits_x + bits_y));
}

template<_DVG_TEMPLATE_DECL_>
void DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::transform(
    const kindr::minimal::QuatTransformationTemplate<float>& transformation) {
  BENCHMARK_BLOCK("SM.TransformLocalMap.TransformDVG");

  // Update transforms
  pose_transformation_ = transformation * pose_transformation_;
  indexing_transformation_ = pose_transformation_.inverse();

  // Transform point clouds in-place
  for(auto centroids : { std::ref(active_centroids_), std::ref(inactive_centroids_)}) {
    for (auto& point : *centroids.get()) {
      point.getVector3fMap() = transformation.transform(point.getVector3fMap());
    }
  }
}

template<_DVG_TEMPLATE_DECL_>
void DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::clear() {
  // Reset transformations.
  pose_transformation_.setIdentity();
  indexing_transformation_.setIdentity();

  // Clear points and voxels.
  active_centroids_->clear();
  inactive_centroids_->clear();
  voxels_.clear();
}

// 体素信息写入日志
template<_DVG_TEMPLATE_DECL_>
void DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::dumpVoxels() const {
  for (const Voxel_& v : voxels_) {
    LOG(INFO) << "Voxel " << uint32_t(v.index) << ": " << v.num_points << " " <<  *(v.centroid);
  }
}

//=================================================================================================
//    DynamicVoxelGrid private methods implementation
//=================================================================================================

// 获取索引，并根据索引排序（升序？）
// 返回：有序的索引点云序列
template<_DVG_TEMPLATE_DECL_>
inline typename DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::IndexedPoints_
DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::indexAndSortPoints_(const InputCloud& points) const {
  IndexedPoints_ indexed_points;
  indexed_points.reserve(points.size());
  // 获取所有点的索引并存储
  // 64位索引，按xyz顺序
  for (const auto& point : points) {
    indexed_points.emplace_back(point, getIndexOf(point));
  }

  // 按照索引升序排序（由于按位存储，所以索引比较顺序为 x y z）
  auto predicate = [](const IndexedPoint_& a, const IndexedPoint_& b) {
    return a.voxel_index < b.voxel_index;
  };
  std::sort(indexed_points.begin(), indexed_points.end(), predicate);

  return indexed_points;
}

// 根据（一个新的）体素信息，更新当前体素列表
// 参数：index体素索引  data记录了新加点的起止迭代器  new_voxels新的体素容器
// 返回：是否为新加入的激活体素
template<_DVG_TEMPLATE_DECL_>
inline bool DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::createVoxel_(
    const IndexT index, const VoxelData_& data,
    std::vector<Voxel_>& new_voxels, VoxelCloud& new_active_centroids,
    VoxelCloud& new_inactive_centroids) {
  VoxelPointT centroid;
  auto centroid_map = centroid.getVector3fMap();
  uint32_t old_points_count = 0u;
  // 新加入当前体素的点的数量
  uint32_t new_points_count = std::distance(data.points_begin, data.points_end);

  // Add contribution from the existing voxel.
  // 计算已存在体素中对质心的贡献
  // 如果新加索引是已存在的，old_voxel才非空
  if (data.old_voxel != nullptr) {
    centroid = *(data.old_voxel->centroid);
    old_points_count = data.old_voxel->num_points;
    if (new_points_count != 0u) {
      centroid_map *= static_cast<float>(old_points_count);
    }
  }
  uint32_t total_points_count = old_points_count + new_points_count;

  // Add contribution from the new points.
  // 添加新点的贡献，从而计算出新的质心
  if (new_points_count != 0u) {
    for (auto it = data.points_begin; it != data.points_end; ++it) {
      centroid_map += it->point.getVector3fMap();
    }
    centroid_map /= static_cast<float>(total_points_count);
  }

  // Save centroid to the correct point cloud.
  // 将质心存到 激活/非激活 质心序列中
  // is_new_voxel是否为新加的激活体素
  VoxelPointT* centroid_pointer;
  bool is_new_voxel = false;
  if (total_points_count >= min_points_per_voxel_) {
    new_active_centroids.push_back(centroid);
    centroid_pointer = &new_active_centroids.back();
    is_new_voxel = (old_points_count < min_points_per_voxel_);
  } else {
    new_inactive_centroids.push_back(centroid);
    centroid_pointer = &new_inactive_centroids.back();
  }

  // 将（体素质心，索引，点的数量）加入新的体素序列
  new_voxels.emplace_back(centroid_pointer, index, total_points_count);
  return is_new_voxel;
}

// 根据输入的to_remove序列，移除target_cloud中对应的点，并返回被成功移除的序列
template<_DVG_TEMPLATE_DECL_>
inline std::vector<bool> DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::removeCentroids_(
    VoxelCloud& target_cloud, std::vector<VoxelPointT*> to_remove) {
  std::vector<bool> is_removed(target_cloud.size(), false);
  if (to_remove.empty()) return is_removed;

  size_t next_removal_index = 0u;
  size_t centroid_index = 0u;

  // Push one more element so that we don't read past the end of the vector.
  to_remove.push_back((VoxelPointT*)0);

  // Remove the required centroids and keep track of their indices.
  // 此处说明质心是有序的？？？？？？？？？
  auto new_end = std::remove_if(target_cloud.begin(), target_cloud.end(),
    [&](VoxelPointT & p){
      const bool remove_p = (&p == to_remove[next_removal_index]);
      if(remove_p) {
        is_removed[centroid_index] = true;
        ++next_removal_index;
      }
      ++centroid_index;
      return remove_p;
  });
  target_cloud.erase(new_end, target_cloud.end());

  // 返回质心是否被移除的标值序列
  return is_removed;
}

} // namespace segmatch

#endif // SEGMATCH_IMPL_DYNAMIC_VOXEL_GRID_HPP_
