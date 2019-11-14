#ifndef SEGMATCH_IMPL_EUCLIDEAN_SEGMENTER_HPP_
#define SEGMATCH_IMPL_EUCLIDEAN_SEGMENTER_HPP_

#define PCL_NO_PRECOMPILE

#include "segmatch/segmenters/euclidean_segmenter.hpp"

#include <laser_slam/benchmarker.hpp>
#include <pcl/segmentation/extract_clusters.h>

#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

// Force the compiler to reuse instantiations provided in euclidean_segmenter.cpp
extern template class EuclideanSegmenter<MapPoint>;

//=================================================================================================
//    EuclideanSegmenter public methods implementation
//=================================================================================================

// min_cluster_size 100
// max_cluster_size 15000
// radius_for_growing 0.2
// ClusteredPointT -> MapPoint
template<typename ClusteredPointT>
EuclideanSegmenter<ClusteredPointT>::EuclideanSegmenter(
    const SegmenterParameters& params)
    : params_(params), min_segment_size_(params.min_cluster_size),
      max_segment_size_(params.max_cluster_size),
      radius_for_growing_(params.radius_for_growing) { }


// 对给定的点云进行聚类，在分节云中写入找到的段
// 如果集群id发生变化，则相应地更新cluster_ids_to_segment_ids映射。
template<typename ClusteredPointT>
void EuclideanSegmenter<ClusteredPointT>::segment(
    const PointNormals& normals, const std::vector<bool>& is_point_modified, ClusteredCloud& cloud,
    PointsNeighborsProvider<MapPoint>& points_neighbors_provider, SegmentedCloud& segmented_cloud,
    std::vector<Id>& cluster_ids_to_segment_ids,
    std::vector<std::pair<Id, Id>>& renamed_segments) {
  BENCHMARK_BLOCK("SM.Worker.Segmenter");

  // Clear segments.
  segmented_cloud.clear();

  std::vector<pcl::PointIndices> cluster_indices;
  // 根据点与点之间的欧氏距离将空间区域聚类
  // 实现：按顺序遍历cloud，kdtree半径搜索，满足条件入队，循环搜索得到一个聚类
  // 如果聚类中的点数量满足要求，加入cluster_indices
  pcl::extractEuclideanClusters<ClusteredPointT>(
      cloud, points_neighbors_provider.getPclSearchObject(), radius_for_growing_, cluster_indices,
      min_segment_size_,max_segment_size_);

  // 各个类别分别加入segmented_cloud
  // segment_id就是按照创建的先后顺序
  for (const auto& point_indices : cluster_indices) {
	// 当前聚类加入SegmentedCloud，若不存在则id自加，若存在则获取已存在id
	// 返回：segment_id
    segmented_cloud.addSegment(point_indices, cloud);
  }

  // 输出聚类类别
  LOG(INFO) << "Segmentation complete. Found " << cluster_indices.size()
        << " clusters ."<< std::endl;
}

} // namespace segmatch

#endif // SEGMATCH_IMPL_EUCLIDEAN_SEGMENTER_HPP_
