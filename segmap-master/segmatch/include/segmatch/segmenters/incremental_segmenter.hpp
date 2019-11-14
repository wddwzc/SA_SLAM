#ifndef SEGMATCH_INCREMENTAL_SEGMENTER_HPP_
#define SEGMATCH_INCREMENTAL_SEGMENTER_HPP_

#include <memory>
#include <stddef.h>
#include <unordered_set>
#include <vector>

#include "segmatch/common.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmenters/region_growing_policy.hpp"
#include "segmatch/segmenters/segmenter.hpp"

namespace segmatch {

// Forward declaration to speed up compilation time.
class SegmentedCloud;

/// \brief Generic incremental region growing segmenter.
/// Extract segments by growing regions in the point cloud according to a specific policy. Allows
/// to update already segmented clouds by segmenting new points only.
// 通用增量式区域增长分割器
// 根据特定的策略，通过点云中的增长区域提取片段
// 只通过分割新的点云来更新已分割点云
template<typename ClusteredPointT, typename PolicyName>
class IncrementalSegmenter : public Segmenter<ClusteredPointT> {
 public:
  typedef pcl::PointCloud<ClusteredPointT> ClusteredCloud;
  // Policy在所给参数中为EuclideanDistance
  typedef RegionGrowingPolicy<PolicyName> Policy;

  static_assert(pcl::traits::has_xyz<ClusteredPointT>::value,
                "IncrementalSegmenter requires ClusteredPointT to contain XYZ coordinates.");

  /// \brief Initializes a new instance of the IncrementalSegmenter class.
  /// \param params The parameters of the segmenter.
  explicit IncrementalSegmenter(const SegmenterParameters& params)
    : params_(params)
	// 0.2
    , search_radius_(params.radius_for_growing)
	// 100
    , min_segment_size_(params.min_cluster_size)
	// 15000
    , max_segment_size_(params.max_cluster_size)
    , policy_params_(Policy::createParameters(params)) {
  }

  /// \brief Cluster the given point cloud, writing the found segments in the segmented cloud. Only
  /// points that are not assigned to a cluster (have the PolicyName::getPointClusterId(point)
  /// equal to zero) will be used as candidate seeds.
  /// If cluster IDs change, the \c cluster_ids_to_segment_ids mapping is updated accordingly.
  /// \param normals The normal vectors of the point cloud. This can be an empty cloud if the
  /// the segmenter doesn't require normals.
  /// \param is_point_modified Indicates for each point if it has been modified such that its
  /// cluster assignment may change.
  /// \param cloud The point cloud that must be segmented.
  /// \param points_neighbors_provider Object providing nearest neighbors information.
  /// \param segmented_cloud Cloud to which the valid segments will be added.
  /// \param cluster_ids_to_segment_ids Mapping between cluster IDs and segment IDs. Cluster
  /// \c i generates segment \c cluster_ids_to_segments_ids[i]. If
  /// \c cluster_ids_to_segments_ids[i] is equal to zero, then the cluster does not contain enough
  /// points to be considered a segment.
  /// \param renamed_segments Vectors containing segments that got a new ID, e.g. after merging
  /// two or more segments. The ordering of the vector represents the sequence of renaming
  /// operations. The first ID in each pair is the renamed segments, the second ID is the new
  /// segment ID.
  // 同虚基类
  // 对所给点云聚类，记录分割块。如果聚类ID改变，聚类ID到分割ID的映射会相应地更新
  // 参数：点云法向量  表示每个点是否修改导致聚类改变  必须分割的点云  提供最近邻信息的对象
  // 参数：将有效分割添加到其中的云  聚类ID和分割ID间的映射  获取到新ID的分割块（合并多个分割后）第一个重命名的，第二个是新的分割ID
  void segment(const PointNormals& normals, const std::vector<bool>& is_point_modified,
               ClusteredCloud& cloud,
               PointsNeighborsProvider<ClusteredPointT>& points_neighbors_provider,
               SegmentedCloud& segmented_cloud, std::vector<Id>& cluster_ids_to_segment_ids,
               std::vector<std::pair<Id, Id>>& renamed_segments) override;

 private:
  typedef uint32_t ClusterId;

  // Helper data structures for discovering and linking partial clusters.
  struct PartialClustersSet {
    ClusterId cluster_id = kUnassignedClusterId;
    Id segment_id = kNoId;
    std::unordered_set<size_t> partial_clusters_indices;
  };
  typedef std::shared_ptr<PartialClustersSet> PartialClustersSetPtr;

  struct PartialCluster {
    PartialCluster() : partial_clusters_set(new PartialClustersSet()) { }
    std::vector<size_t> point_indices;
    PartialClustersSetPtr partial_clusters_set;
  };
  typedef std::vector<PartialCluster> PartialClusters;

  // Determine the ID of a segment created by the merging of segments with IDs \c id_1 and \c id_2.
  // In case two segments are merged, the first ID is the discarded ID, and the second ID is the
  // ID used for the merged segment. In the other cases the first ID specifies if one of the
  // segments was kInvId or kNoId.
  // 决定由id1和id2分割块和并得到的分割块的ID，丢弃id1使用id2，如果kInvId或kNoId，指定id1
  std::pair<Id, Id> mergeSegmentIds(Id id_1, Id id_2) const;

  // Specifies that two partial clusters belong to the same cluster and must be linked. As a result
  // of this operation, both partial clusters will point to the same partial clusters set.
  // 指定两个局部聚类属于同一聚类且必须连接
  void linkPartialClusters(size_t partial_cluster_1_index, size_t partial_cluster_2_index,
                           PartialClusters& partial_clusters,
                           std::vector<std::pair<Id, Id>>& renamed_segments) const;

  // Grows a region starting from the specified seed point. This finds all the new points belonging
  // to the same cluster and possibly links to existing clusters. The resulting partial cluster is
  // added to the \c partial_clusters vector.
  // 从指定的种子点开始生长区域，找到所有属于同一聚类且连接到已存在聚类的新点，得到局部聚类加入局部聚类序列
  void growRegionFromSeed(const PointNormals& normals, const ClusteredCloud& cloud,
                          PointsNeighborsProvider<ClusteredPointT>& points_neighbors_provider,
                          size_t seed_index, std::vector<bool>& processed,
                          PartialClusters& partial_clusters,
                          std::vector<std::pair<Id, Id>>& renamed_segments) const;

  // Clusters a point cloud. Only new or modified points are used as seeds.
  // 对点云聚类，只有新的或者修改的点被用作种子
  void growRegions(const PointNormals& normals, const std::vector<bool>& is_point_modified,
                   const std::vector<Id>& cluster_ids_to_segment_ids, ClusteredCloud& cloud,
                   PointsNeighborsProvider<ClusteredPointT>& points_neighbors_provider,
                   PartialClusters& partial_clusters,
                   std::vector<std::pair<Id, Id>>& renamed_segments) const;

  // Assign cluster indices to the sets of partial clusters, so that linked clusters have the same
  // cluster index and clusters use contiguous indices starting from 1. Returns the total number of
  // clusters.
  // 聚类 改为 团簇
  // 将聚类索引对应到局部聚类集合，连在一起的团簇由先沟通的索引，团簇使用从1开始的连续索引，返回团簇数量
  size_t assignClusterIndices(const PartialClusters& partial_clusters) const;

  // Write the cluster indices in the point cloud so that they can be reused in future
  // segmentations.
  // 将团簇索引写入点云，以便于在分割时可复用
  void writeClusterIndicesToCloud(const PartialClusters& partial_clusters,
                                  ClusteredCloud& cloud) const;

  // Adds the segments to a segmented cloud and updates \c cluster_ids_to_segment_ids to reflect
  // the new mapping between cluster IDs and segment IDs.
  // 将分割块加入分割点云，更新id映射
  void addSegmentsToSegmentedCloud(const ClusteredCloud& cloud,
                                   const PartialClusters& partial_clusters, size_t num_clusters,
                                   std::vector<Id>& cluster_ids_to_segment_ids,
                                   SegmentedCloud& segmented_cloud) const;

  // Get the total number of points contained in the cluster of which the partial cluster at index
  // \i partial_cluster_index is part.
  // 获取partial_cluster_index对应的局部团簇包含点的数量
  size_t getClusterSize(const PartialClusters& partial_clusters,
                        size_t partial_cluster_index) const;

  // Get the indices of the points contained in the cluster of which the partial cluster at index
  // \i partial_cluster_index is part.
  // 获取partial_cluster_index对应的团簇中点的索引
  std::vector<int> getClusterIndices(const PartialClusters& partial_clusters,
                                     size_t partial_cluster_index) const;

  // Determines if a point is assigned to a cluster or not.
  // 决定点是否关联到团簇
  bool isPointAssignedToCluster(const ClusteredPointT& point) const noexcept;

  // Gets the cluster ID of a point.
  // 获取一个点的团簇ID
  ClusterId getClusterId(const ClusteredPointT& point) const noexcept;

  // Sets the cluster ID of a point.
  // 
  void setClusterId(ClusteredPointT& point, ClusterId cluster_id) const noexcept;

  // Segmenter settings.
  SegmenterParameters params_;
  const double search_radius_;
  const int min_segment_size_;
  const int max_segment_size_;
  typename Policy::PolicyParameters policy_params_;

  static constexpr ClusterId kUnassignedClusterId = 0u;
}; // class IncrementalSegmenter

} // namespace segmatch

#endif // SEGMATCH_INCREMENTAL_SEGMENTER_HPP_
