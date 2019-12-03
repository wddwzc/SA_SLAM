#ifndef SEGMATCH_IMPL_INCREMENTAL_SEGMENTER_HPP_
#define SEGMATCH_IMPL_INCREMENTAL_SEGMENTER_HPP_

#include "segmatch/segmenters/incremental_segmenter.hpp"

#include <algorithm>

#include <laser_slam/benchmarker.hpp>

#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

// Force the compiler to reuse instantiations provided in incremental_segmenter.cpp
extern template class IncrementalSegmenter<MapPoint, EuclideanDistance>;
extern template class IncrementalSegmenter<MapPoint, SmoothnessConstraints>;

//=================================================================================================
//    IncrementalSegmenter public methods implementation
//=================================================================================================

template<typename ClusteredPointT, typename PolicyName>
void IncrementalSegmenter<ClusteredPointT, PolicyName>::segment(
    const PointNormals& normals, const std::vector<bool>& is_point_modified, ClusteredCloud& cloud,
    PointsNeighborsProvider<ClusteredPointT>& points_neighbors_provider,
    SegmentedCloud& segmented_cloud, std::vector<Id>& cluster_ids_to_segment_ids,
    std::vector<std::pair<Id, Id>>& renamed_segments) {
  BENCHMARK_BLOCK("SM.Worker.Segmenter");
  renamed_segments.clear();
  
  // Build partial cluster sets for the old clusters.
  // 创建局部团簇
  PartialClusters partial_clusters(cluster_ids_to_segment_ids.size());
  for (size_t i = 0u; i < partial_clusters.size(); i++) {
	// partial_clusters_indices保存部分聚类索引
    partial_clusters[i].partial_clusters_set->partial_clusters_indices.insert(i);
	// segment_id保存分割id
    partial_clusters[i].partial_clusters_set->segment_id = cluster_ids_to_segment_ids[i];
  }

  // Find old clusters and new partial clusters.
  // 找到旧聚类和新的局部聚类
  // 添加新的是改变point_indices，合并是改变set
  growRegions(normals, is_point_modified, cluster_ids_to_segment_ids, cloud,
              points_neighbors_provider, partial_clusters, renamed_segments);

  // Compute and write cluster indices.
  // 计算并分配聚类索引
  const size_t num_clusters = assignClusterIndices(partial_clusters);
  writeClusterIndicesToCloud(partial_clusters, cloud);

  // Extract the valid segment and add them to the segmented cloud.
  addSegmentsToSegmentedCloud(cloud, partial_clusters, num_clusters, cluster_ids_to_segment_ids,
                              segmented_cloud);
}

//=================================================================================================
//    IncrementalSegmenter private methods implementation
//=================================================================================================

// 合并分割块id
// 返回：{大，小} {无效，有效}
template<typename ClusteredPointT, typename PolicyName>
inline std::pair<Id, Id> IncrementalSegmenter<ClusteredPointT, PolicyName>::mergeSegmentIds(
    const Id id_1, const Id id_2) const {
  if (id_1 == kInvId || id_2 == kInvId) {
    // Invalidated segments stay invalid.
    return  { kInvId, kInvId };
  } else if (id_1 == kNoId) {
    // In case one cluster doesn't belong to a segment keep the only segment ID (if any). No
    // renaming necessary.
    return { kNoId, id_2 };
  } else if (id_2 == kNoId) {
    // In case one cluster doesn't belong to a segment keep the only segment ID (if any). No
    // renaming necessary.
    return { kNoId, id_1 };
  } else {
    // Otherwise take the minimum segment ID (the one that has been around for longer). The segment
    // with maximum ID is renamed to the minimum ID.
    return { std::max(id_1, id_2), std::min(id_1, id_2) };
  }
}

// 实现两个部分集群的合并，id保留较小值
template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::linkPartialClusters(
    const size_t partial_cluster_1_index, const size_t partial_cluster_2_index,
    PartialClusters& partial_clusters, std::vector<std::pair<Id, Id>>& renamed_segments) const {
  // Get pointers to the partial cluster sets
  PartialClustersSetPtr set_1 = partial_clusters[partial_cluster_1_index].partial_clusters_set;
  PartialClustersSetPtr set_2 = partial_clusters[partial_cluster_2_index].partial_clusters_set;

  // Both partial clusters belong to the same set. Nothing to do.
  if (set_1 == set_2) return;

  // Swap the partial cluster indices if it makes the merge operation faster.
  // 交换部分集群索引，如果这样可以加快合并操作
  if (set_1->partial_clusters_indices.size() < set_2->partial_clusters_indices.size()) {
    std::swap(set_1, set_2);
  }

  // Move the linked indices from set_2 to set_1 and determine the segment ID.
  // 将set2合并到set1
  set_1->partial_clusters_indices.insert(set_2->partial_clusters_indices.begin(),
                                         set_2->partial_clusters_indices.end());
  Id old_segment_id;
  // set_1->segment_id保留较小的id
  std::tie(old_segment_id, set_1->segment_id) = mergeSegmentIds(set_1->segment_id,
                                                                set_2->segment_id);

  // Detect if a segment renaming happened
  // 如果没有无效id，才会有映射
  if (old_segment_id != kNoId && old_segment_id != kInvId)
    renamed_segments.push_back({ old_segment_id, set_1->segment_id });

  // Update all partial clusters contained in set_2 so that they point to set_1.
  // 更新set_2中包含的所有部分集群，使它们指向set_1
  for (const auto partial_cluster_index : set_2->partial_clusters_indices) {
    partial_clusters[partial_cluster_index].partial_clusters_set = set_1;
  }
}

// 从种子进行区域生长
// 输入参数：normals, cloud, points_neighbors_provider, i, processed, partial_clusters, renamed_segments
// normals 法向量
// cloud 点云
// points_neighbors_provider 邻域提供
// seed_index 种子索引
// processed 处理过的种子标志
// partial_clusters 局部聚类

// 如果生长到的位置是已分配的，直接进行合并
// 如果是未分配的，则作为种子，继续生长
template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::growRegionFromSeed(
    const PointNormals& normals, const ClusteredCloud& cloud,
    PointsNeighborsProvider<ClusteredPointT>& points_neighbors_provider, const size_t seed_index,
    std::vector<bool>& processed, PartialClusters& partial_clusters,
    std::vector<std::pair<Id, Id>>& renamed_segments) const {
  // Create a new partial cluster.
  partial_clusters.emplace_back();
  PartialCluster& partial_cluster = partial_clusters.back();
  size_t partial_cluster_id = partial_clusters.size() - 1u;
  partial_cluster.partial_clusters_set = std::make_shared<PartialClustersSet>();
  partial_cluster.partial_clusters_set->partial_clusters_indices.insert(partial_cluster_id);

  // Initialize the seeds queue.
  std::vector<size_t>& region_indices = partial_cluster.point_indices;
  std::vector<size_t> seed_queue;
  size_t current_seed_index = 0u;
  seed_queue.push_back(seed_index);
  region_indices.push_back(seed_index);

  // Search for neighbors until there are no more seeds.
  while (current_seed_index < seed_queue.size()) {
    // Search for points around the seed.
	// 此处搜索半径参数为0.2
    std::vector<int> neighbors_indices = points_neighbors_provider.getNeighborsOf(
        seed_queue[current_seed_index], search_radius_);

    // Decide on which points should we continue the search and if we have to link partial
    // clusters.
	// 判断哪些点要继续搜索
	// 对未处理的点进行搜索
    for (const auto neighbor_index : neighbors_indices) {
	  // 在Euclidean方法下，canGrowToPoint为true
      if (neighbor_index != -1 && Policy::canGrowToPoint(
          policy_params_, normals, seed_queue[current_seed_index], neighbor_index)) {
		
		// 判断point.ed_cluster_id != 0
        if (isPointAssignedToCluster(cloud[neighbor_index])) {
          // If the search reaches an existing cluster we link to its partial clusters set.
		  // partial_cluster_id != point.ed_cluster_id
          if (partial_cluster_id != getClusterId(cloud[neighbor_index])) {
			// 实现两个部分集群的合并，id保留较小值
            linkPartialClusters(partial_cluster_id, getClusterId(cloud[neighbor_index]),
                                partial_clusters, renamed_segments);
          }
        } else if (!processed[neighbor_index]) {
          // Determine if the point can be used as seed for the region.
		  // 如果是未处理的点，就判断是否可以作为种子
          if (Policy::canPointBeSeed(policy_params_, normals, neighbor_index)) {
            seed_queue.push_back(neighbor_index);
          }
          // Assign the point to the current partial cluster.
		  // 将未处理点添加到point_indices中
          region_indices.push_back(neighbor_index);
          processed[neighbor_index] = true;
        }
      }
    }
    ++current_seed_index;
  }
}

// 区域生长
// 疑问：ed_cluster_id在哪里赋值的？？？？？？？？？？？？？？？？？
template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::growRegions(
    const PointNormals& normals, const std::vector<bool>& is_point_modified,
    const std::vector<Id>& cluster_ids_to_segment_ids, ClusteredCloud& cloud,
    PointsNeighborsProvider<ClusteredPointT>& points_neighbors_provider,
    PartialClusters& partial_clusters, std::vector<std::pair<Id, Id>>& renamed_segments) const {
  BENCHMARK_BLOCK("SM.Worker.Segmenter.GrowRegions");

  std::vector<bool> processed(cloud.size(), false);
  std::vector<size_t> new_points_indices;
  new_points_indices.reserve(cloud.size());

  for (size_t i = 0u; i < cloud.size(); ++i) {
	// cloud指定了对应聚类分割块，通过分割点云索引不等于0判断
    if (isPointAssignedToCluster(cloud[i])) {
      // No need to cluster points that are already assigned.
	  // 已经分配到聚类的点就不需要在处理，直接添加即可
	  // 根据ed_cluster_id将点云添加到对应部分聚类的点序列中
      partial_clusters[getClusterId(cloud[i])].point_indices.push_back(i);
    } else if (Policy::canPointBeSeed(policy_params_, normals, i)) { // Euclidean中，返回true
      new_points_indices.emplace_back(i);
    }
  }

  // Prepare the seed indices.
  // 排序
  Policy::prepareSeedIndices(normals, new_points_indices.begin(), new_points_indices.end());

  // Process the new points.
  // TODO: The current implementation ignores any change in the normal/curvature of a point,
  // ignoring cases in which changes in the properties of a point would lead to different
  // clustering decisions. It would be nice to add segmentation policies covering this case.
  // 处理新加入的点，new_points_indices是新加点在cloud中的索引的列表
  for (const auto i : new_points_indices) {
    if (!processed[i]) {
      // Mark the point as processed and grow the cluster starting from it.
      processed[i] = true;
      growRegionFromSeed(normals, cloud, points_neighbors_provider, i, processed, partial_clusters,
                         renamed_segments);
    }
  }
}

// 分配聚类索引，从1开始
template<typename ClusteredPointT, typename PolicyName>
inline size_t IncrementalSegmenter<ClusteredPointT, PolicyName>::assignClusterIndices(
    const PartialClusters& partial_clusters) const {
  BENCHMARK_BLOCK("SM.Worker.Segmenter.AssignClusterIndices");

  // Assign cluster IDs.
  ClusterId next_cluster_id = 1u;
  for (const auto& partial_cluster : partial_clusters) {
    const PartialClustersSetPtr& partial_clusters_set = partial_cluster.partial_clusters_set;
    if (!partial_cluster.point_indices.empty() &&
        partial_clusters_set->cluster_id == kUnassignedClusterId) {
      // Assign a cluster index only if the set didn't get one yet and the partial cluster
      // contains at least one point.
	  // 在point_indices非空 而 set没有分配到某个聚类时
	  // 也就是没有进行新旧合并，该局部聚类是第一次出现？？？？存疑
	  // 给其分配聚类id
      partial_clusters_set->cluster_id = next_cluster_id;
      ++next_cluster_id;
    }
  }

  return static_cast<size_t>(next_cluster_id);
}

// 将聚类索引写入点云
template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::writeClusterIndicesToCloud(
    const PartialClusters& partial_clusters, ClusteredCloud& cloud) const {
  BENCHMARK_BLOCK("SM.Worker.Segmenter.WriteClusterIndices");

  // Write cluster IDs in the point cloud.
  // 给点云写入聚类id，即PointExtend中的ed_cluster_id或sc_cluster_id
  for (const auto& partial_cluster : partial_clusters) {
    for (const auto point_id : partial_cluster.point_indices) {
      setClusterId(cloud[point_id], partial_cluster.partial_clusters_set->cluster_id);
    }
  }
}

// 将分割添加到segmented_cloud
template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::addSegmentsToSegmentedCloud(
    const ClusteredCloud& cloud, const PartialClusters& partial_clusters,
    const size_t num_clusters, std::vector<Id>& cluster_ids_to_segment_ids,
    SegmentedCloud& segmented_cloud) const {
  BENCHMARK_BLOCK("SM.Worker.Segmenter.AddSegments");
  BENCHMARK_RECORD_VALUE("SM.NumClusters", num_clusters);

  // Initially all clusters don't have a segment ID.
  // 建立cluster id到segment id的映射
  cluster_ids_to_segment_ids = std::vector<Id>(num_clusters, kUnassignedId);
  if (!cluster_ids_to_segment_ids.empty()) cluster_ids_to_segment_ids[0] = kNoId;

  std::vector<Id> segment_ids_to_keep;

  // 遍历各部分聚类
  for (size_t i = 0u; i < partial_clusters.size(); i++) {
    const PartialClustersSetPtr& partial_clusters_set = partial_clusters[i].partial_clusters_set;
    const ClusterId cluster_id = partial_clusters_set->cluster_id;

    // Only process clusters once.
	// 如果当前聚类不是未分配的，就不要进一步处理了，直接跳过
    if (cluster_ids_to_segment_ids[cluster_id] != kUnassignedId) continue;

    const Id old_segment_id = partial_clusters_set->segment_id;
	// 跳过无效segment id
    if (old_segment_id == kInvId) {
      // Skip invalidated segments
      cluster_ids_to_segment_ids[cluster_id] = kInvId;
    } else {
	// 如果segment id有效
	  // 获取聚类点尺寸
      const size_t points_in_cluster = getClusterSize(partial_clusters, i);
	  // 参数中max_segment_size_未15000
      if (points_in_cluster > max_segment_size_) {
        // Invalidate segments with too many points.
		// 点太多，所以无效
        cluster_ids_to_segment_ids[cluster_id] = kInvId;
      } else if (old_segment_id != kNoId || points_in_cluster >= min_segment_size_) {
	  // 最少分割点数量100 或 之前已存在的segment
        // Create the segment, reusing the previous segment ID if present.
		// 创建segment或重用已有的segment ID
        pcl::PointIndices point_indices;
		// 获取所有关联点云
        point_indices.indices = getClusterIndices(partial_clusters, i);
		// 将聚类点云加入segmented_cloud
        cluster_ids_to_segment_ids[cluster_id] = segmented_cloud.addSegment(
            point_indices, cloud, old_segment_id);

		// 需要保留的segment_ids
        segment_ids_to_keep.push_back(cluster_ids_to_segment_ids[cluster_id]);
        BENCHMARK_RECORD_VALUE("SM.SegmentSize", point_indices.indices.size());
      } else {
        // The cluster doesn't have enough points, don't assign a segment yet.
        cluster_ids_to_segment_ids[cluster_id] = kNoId;
      }
    }
  }

  // Delete the segments that we did not keep.
  segmented_cloud.deleteSegmentsExcept(segment_ids_to_keep);
}

// 获取聚类尺寸，所有关联聚类
template<typename ClusteredPointT, typename PolicyName>
inline size_t IncrementalSegmenter<ClusteredPointT, PolicyName>::getClusterSize(
    const PartialClusters& partial_clusters, const size_t partial_cluster_index) const {
  size_t points_in_cluster = 0u;
  const PartialClustersSetPtr& partial_clusters_set =
      partial_clusters[partial_cluster_index].partial_clusters_set;
  for (const auto linked_partial_cluster_index : partial_clusters_set->partial_clusters_indices) {
    points_in_cluster += partial_clusters[linked_partial_cluster_index].point_indices.size();
  }
  return points_in_cluster;
}

// 获取与当前索引关联的所有部分聚类中点的索引
template<typename ClusteredPointT, typename PolicyName>
inline std::vector<int> IncrementalSegmenter<ClusteredPointT, PolicyName>::getClusterIndices(
    const PartialClusters& partial_clusters, const size_t partial_cluster_index) const {
  const PartialClustersSetPtr& partial_clusters_set =
      partial_clusters[partial_cluster_index].partial_clusters_set;

  std::vector<int> point_indices;
  point_indices.reserve(getClusterSize(partial_clusters, partial_cluster_index));
  for (const auto linked_partial_cluster_index : partial_clusters_set->partial_clusters_indices) {
    point_indices.insert(point_indices.end(),
                         partial_clusters[linked_partial_cluster_index].point_indices.begin(),
                         partial_clusters[linked_partial_cluster_index].point_indices.end());
  }
  return point_indices;
}

// point.ed_cluster_id != 0
template<typename ClusteredPointT, typename PolicyName>
inline bool IncrementalSegmenter<ClusteredPointT, PolicyName>::isPointAssignedToCluster(
    const ClusteredPointT& point) const noexcept {
  return getClusterId(point) != 0u;
}

// 返回的是point.ed_cluster_id
template<typename ClusteredPointT, typename PolicyName>
inline typename IncrementalSegmenter<ClusteredPointT, PolicyName>::ClusterId
IncrementalSegmenter<ClusteredPointT, PolicyName>::getClusterId(
    const ClusteredPointT& point) const noexcept{
  return Policy::getPointClusterId(point);
}

// 设置聚类id，即给ed_cluster_id或sc_cluster_id赋值
template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::setClusterId(
    ClusteredPointT& point, const ClusterId cluster_id) const noexcept {
  Policy::setPointClusterId(point, cluster_id);
}

} // namespace segmatch

#endif // SEGMATCH_IMPL_INCREMENTAL_SEGMENTER_HPP_
