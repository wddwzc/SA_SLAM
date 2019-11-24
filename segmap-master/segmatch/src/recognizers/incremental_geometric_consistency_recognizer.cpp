#include "segmatch/recognizers/incremental_geometric_consistency_recognizer.hpp"

#include <limits>
#include <vector>

#include <glog/logging.h>
#include <laser_slam/benchmarker.hpp>

#include "segmatch/common.hpp"
#include "segmatch/recognizers/graph_utilities.hpp"
#include "segmatch/recognizers/matches_partitioner.hpp"

namespace segmatch {

constexpr size_t IncrementalGeometricConsistencyRecognizer::kNoMatchIndex_;
constexpr size_t IncrementalGeometricConsistencyRecognizer::kNoCacheSlotIndex_;

// 初始化
// 注意初始化列表
IncrementalGeometricConsistencyRecognizer::IncrementalGeometricConsistencyRecognizer(
    const GeometricConsistencyParams& params, const float max_model_radius) noexcept
  : GraphBasedGeometricConsistencyRecognizer(params)
  // 此处max_model_radius为50
  // resolution为0.4或0.6
  , max_consistency_distance_(max_model_radius * 2.0 + params.resolution)
  // max_consistency_distance_for_caching为3
  , max_consistency_distance_for_caching_(
      params.max_consistency_distance_for_caching + params.resolution)
  , half_max_consistency_distance_for_caching_(params.max_consistency_distance_for_caching * 0.5f) {
}

// 处理缓存中已存在的预测匹配，清理旧条目，找到一致性并添加到一致性图
// 参数： cached_matches_locations  match的索引，缓存的索引
// cache_slot_index_to_match_index  从缓存到match的索引映射
inline void IncrementalGeometricConsistencyRecognizer::processCachedMatches(
    const PairwiseMatches& predicted_matches,
    const std::vector<MatchLocations>& cached_matches_locations,
    const std::vector<size_t>& cache_slot_index_to_match_index,
    std::unordered_map<IdPair, size_t, IdPairHash>& new_cache_slot_indices,
    ConsistencyGraph& consistency_graph) {
  BENCHMARK_BLOCK("SM.Worker.Recognition.BuildConsistencyGraph.CachedMatches");

  // Recompute consistency information of cached elements where necessary.
  // 必要时重新计算缓存元素的一致性信息。
  size_t num_consistency_tests = 0u;
  for (const auto& cached_match_locations : cached_matches_locations) {
	// match 匹配对
	// match_cache 缓存的匹配对
	// new_cache_slot_indices  存储match的ids 和 缓存索引
    const PairwiseMatch& match = predicted_matches[cached_match_locations.match_index];
    MatchCacheSlot& match_cache = matches_cache_[cached_match_locations.cache_slot_index];
    new_cache_slot_indices.emplace(match.ids_, cached_match_locations.cache_slot_index);

    // For each cached element, get rid of any reference to matches that do not exist anymore and
    // add consistent pairs to the consistency graph.
	// 对于每个缓存的元素，删除不再存在的匹配项的所有引用，并向一致性图中添加一致性对。
    std::vector<size_t> new_candidate_consistent_matches;
    new_candidate_consistent_matches.reserve(match_cache.candidate_consistent_matches.size());
    for (const size_t candidate_cache_slot_index : match_cache.candidate_consistent_matches) {
      const size_t match_2_index = cache_slot_index_to_match_index[candidate_cache_slot_index];
	  // 不等于kNoMatchIndex_，说明从缓存中移除了
      if (match_2_index != kNoMatchIndex_) {
        const PairwiseMatch& match_2 = predicted_matches[match_2_index];
		// match
        float consistency_distance = computeConsistencyDistance(match, match_2,
                                                                max_consistency_distance_);
        ++num_consistency_tests;

        // If the matches are close enough, cache them as candidate consistent matches.
		// 如果匹配足够接近，缓存作为候选一致匹配（阈值约为100）
        if (consistency_distance <= max_consistency_distance_) {
          new_candidate_consistent_matches.emplace_back(candidate_cache_slot_index);

          // If the matches are consistent, add and edge to the consistency graph
		  // 如果匹配一致，将边添加到一致性图（阈值为0.4或0.6）
          if (consistency_distance <= params_.resolution)
            boost::add_edge(match_2_index, cached_match_locations.match_index,
                            consistency_graph);
        }
      }
    }
    match_cache.candidate_consistent_matches = std::move(new_candidate_consistent_matches);
  }
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.TestedCachedPairs",
                         num_consistency_tests);
}

// 处理缓存中不存在的预测匹配，找到一致性并添加到一致性图
// 参数：
// predicted_matches  预测匹配
// free_cache_slot_indices  被释放掉不再使用的缓存槽的索引
// match_index_to_cache_slot_index  match索引到cache的映射
// new_cache_slot_indices  新的缓存索引
// consistency_graph  一致性图
inline void IncrementalGeometricConsistencyRecognizer::processNewMatches(
    const PairwiseMatches& predicted_matches,
    const std::vector<size_t>& free_cache_slot_indices,
    std::vector<size_t>& match_index_to_cache_slot_index,
    std::unordered_map<IdPair, size_t, IdPairHash>& new_cache_slot_indices,
    ConsistencyGraph& consistency_graph) {
  BENCHMARK_BLOCK("SM.Worker.Recognition.BuildConsistencyGraph.NewMatches");

  // Partition the matches in a grid by the position of the scene points. The size of the
  // partitions is greater or equal the size of the model. This way we can safely assume that, if
  // the model is actually present in the scene, all matches will be contained in a 2x2 group of
  // adjacent partitions.
  
  // 根据场景点的位置在网格中划分匹配项，分区的尺寸大于等于模型
  // 这样我们就可以放心地假设，如果这个模型实际上出现在场景中，所有匹配项将包含在一个2x2相邻的分区组中。
  BENCHMARK_START("SM.Worker.Recognition.BuildConsistencyGraph.Partitioning");
  MatchesGridPartitioning<PartitionData> partitioning =
      MatchesPartitioner::computeGridPartitioning<PartitionData>(predicted_matches,
                                                                 max_consistency_distance_);
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.NumPartitions",
                         partitioning.getHeight() * partitioning.getWidth());
  BENCHMARK_STOP("SM.Worker.Recognition.BuildConsistencyGraph.Partitioning");

  // Find all possible consistency within a partition and within neighbor partitions.
  size_t num_consistency_tests = 0u;
  size_t next_slot_index_position = 0u;
  for (size_t i = 0; i < partitioning.getHeight(); ++i) {
    for (size_t j = 0; j < partitioning.getWidth(); ++j) {
      for (const auto match_index : partitioning(i, j).match_indices) {
        // Only process new matches.
        if (match_index_to_cache_slot_index[match_index] != kNoCacheSlotIndex_) continue;
        const PairwiseMatch& match = predicted_matches[match_index];

        // Get a free cache slot and insert the match.
        const size_t cache_slot_index = free_cache_slot_indices[next_slot_index_position];
        ++next_slot_index_position;
        MatchCacheSlot& match_cache = matches_cache_[cache_slot_index];
        match_cache.candidate_consistent_matches.clear();
        match_cache.candidate_consistent_matches.reserve(matches_cache_.size() - 1u);
        match_cache.centroids_at_caching = match.centroids_;
        new_cache_slot_indices.emplace(match.ids_, cache_slot_index);

        // Test consistencies between the current match and the cached matches in the neighbor
        // partitions.
        for (size_t k = static_cast<size_t>(std::max(0, static_cast<int>(i) - 1));
             k <= std::min(partitioning.getHeight() - 1u, i + 1u); ++k) {
          for (size_t l = static_cast<size_t>(std::max(0, static_cast<int>(j) - 1));
               l <= std::min(partitioning.getWidth() - 1u, j + 1u); ++l) {
            for (const auto match_2_index : partitioning(k, l).match_indices) {
              // Only compare to matches already present in the cache
              if (match_index_to_cache_slot_index[match_2_index] != kNoCacheSlotIndex_) {
                const PairwiseMatch& match_2 = predicted_matches[match_2_index];
                float consistency_distance = computeConsistencyDistance(match, match_2,
                                                                        max_consistency_distance_);
                ++num_consistency_tests;

                // If the matches are close enough, cache them as candidate consistent matches.
                if (consistency_distance <= max_consistency_distance_for_caching_) {
                  match_cache.candidate_consistent_matches.emplace_back(
                      match_index_to_cache_slot_index[match_2_index]);
                  // If the matches are consistent, add an edge to the consistency graph.
                  if (consistency_distance <= params_.resolution)
                    boost::add_edge(match_index, match_2_index, consistency_graph);
                }
              }
            }
          }
        }

        match_index_to_cache_slot_index[match_index] = cache_slot_index;
      }
    }
  }
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.TestedNewPairs",
                           num_consistency_tests);
}

bool IncrementalGeometricConsistencyRecognizer::mustRemoveFromCache(
    const PairwiseMatch& match, const size_t cache_slot_index) {
  const MatchCacheSlot& match_cache = matches_cache_[cache_slot_index];
  pcl::Vector3fMapConst model_centroid = match.centroids_.first.getVector3fMap();
  pcl::Vector3fMapConst scene_centroid = match.centroids_.second.getVector3fMap();
  pcl::Vector3fMapConst model_centroid_at_caching =
      match_cache.centroids_at_caching.first.getVector3fMap();
  pcl::Vector3fMapConst scene_centroid_at_caching =
      match_cache.centroids_at_caching.second.getVector3fMap();

  // Since checking the change in consistency distance for every match pair would be too expensive,
  // we the responsibility of the check on both matches. If the centroids of a match move by half
  // the maximum distance allowed, then the cached information are invalidated independently of the
  // changes of the other matches.
  // 由于检查每个匹配对的一致性距离的变化开销过大，我们只检查两个匹配。
  // 如果匹配的质心移动了允许的最大距离的一半，则高速缓存的信息无效，而与其他匹配的变化无关。
  const float model_displacement = (model_centroid - model_centroid_at_caching).norm();
  const float scene_displacement = (scene_centroid - scene_centroid_at_caching).norm();
  // 该参数大概为1.5
  return model_displacement + scene_displacement >= half_max_consistency_distance_for_caching_;
}

inline IncrementalGeometricConsistencyRecognizer::ConsistencyGraph
IncrementalGeometricConsistencyRecognizer::buildConsistencyGraph(
    const PairwiseMatches& predicted_matches) {
  BENCHMARK_BLOCK("SM.Worker.Recognition.BuildConsistencyGraph");

  // Resize the cache to fit the new matches.
  if (predicted_matches.size() > matches_cache_.size())
    matches_cache_.resize(predicted_matches.size());
  std::vector<size_t> cache_slot_index_to_match_index(matches_cache_.size(), kNoMatchIndex_);

  // Identify which matches have cached information.
  // 识别哪些匹配项已经缓存了信息
  size_t invalidated_cached_matches = 0u;
  std::vector<MatchLocations> cached_matches_locations;
  cached_matches_locations.reserve(predicted_matches.size());
  std::vector<size_t> match_index_to_cache_slot_index(predicted_matches.size(),
                                                      kNoCacheSlotIndex_);
  for (size_t i = 0u; i < predicted_matches.size(); ++i) {
    const auto cached_info_it = cache_slot_indices_.find(predicted_matches[i].ids_);
    if (cached_info_it != cache_slot_indices_.end()) {
      // If a centroid moved by more than the allowed distance, we need to invalidate the cached
      // information and threat the match as new.
	  // 如果质心移动超过允许的距离，我们需要使缓存信息失效 和威胁比赛如新。
      if (mustRemoveFromCache(predicted_matches[i], cached_info_it->second)) {
        ++invalidated_cached_matches;
      } else {
        cached_matches_locations.emplace_back(i, cached_info_it->second);
        cache_slot_index_to_match_index[cached_info_it->second] = i;
        match_index_to_cache_slot_index[i] = cached_info_it->second;
      }
    }
  }
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.InvalidatedMatches",
                         invalidated_cached_matches);
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.CachedMatches",
                         cached_matches_locations.size());

  // Collect indices of the cache slots that are not used anymore.
  // 收集不再使用的缓存槽的索引
  // cache_slot_index_to_match_index初始值为kNoMatchIndex_，判断缓存无效时不对其进行赋值操作
  std::vector<size_t> free_cache_slot_indices;
  free_cache_slot_indices.reserve(matches_cache_.size());
  for (size_t i = 0u; i < matches_cache_.size(); ++i) {
    if (cache_slot_index_to_match_index[i] == kNoMatchIndex_)
      free_cache_slot_indices.push_back(i);
  }

  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.TotalMatches", predicted_matches.size());
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.CachedMatches", cached_matches_locations.size());

  // Build the consistency graph.
  // 构建一致性图
  // ConsistencyGraph -> boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>
  ConsistencyGraph consistency_graph(predicted_matches.size());
  std::unordered_map<IdPair, size_t, IdPairHash> new_cache_slot_indices;
  new_cache_slot_indices.reserve(predicted_matches.size());
  // cached_matches_locations  一是candidate_consistent_matches，二是centroids_at_caching
  // cache_slot_index_to_match_index  用kNoMatchIndex_初始化，cache到match的索引映射
  // match_index_to_cache_slot_index  用kNoMatchIndex_初始化，match的索引到cache的映射
  // cache_slot_indices_  一IdPair，二size_t
  processCachedMatches(predicted_matches, cached_matches_locations,
                       cache_slot_index_to_match_index, new_cache_slot_indices, consistency_graph);
  processNewMatches(predicted_matches, free_cache_slot_indices, match_index_to_cache_slot_index,
                    new_cache_slot_indices, consistency_graph);

  // Use the new mapping between match IDs and cache slots.
  cache_slot_indices_ = std::move(new_cache_slot_indices);
  return consistency_graph;
}

// 计算一致性距离（scene_dis与model_dis差值）
// 参数： max_target_distance=50*2
// 这样是否就没考虑旋转的问题？？？？？？？？？
inline float IncrementalGeometricConsistencyRecognizer::computeConsistencyDistance(
    const PairwiseMatch& first_match, const PairwiseMatch& second_match,
    const float max_target_distance) const {
  // Get the centroids of the matched segments.
  // 获取匹配分割的质心
  pcl::Vector3fMapConst model_point_1 = first_match.centroids_.first.getVector3fMap();
  pcl::Vector3fMapConst scene_point_1 = first_match.centroids_.second.getVector3fMap();
  pcl::Vector3fMapConst model_point_2 = second_match.centroids_.first.getVector3fMap();
  pcl::Vector3fMapConst scene_point_2 = second_match.centroids_.second.getVector3fMap();

  // If the keypoints are so far away in the scene so that they can not fit in the model together,
  // the matches will not be consistent even if the centroids in the model move.
  // 如果场景（target）中的关键点离得太远，以至于它们不能一起放入模型中，
  // 即使模型（source）中的质心移动，匹配也不会是一致的
  const float scene_distance = (scene_point_1 - scene_point_2).norm();
  if (scene_distance > max_target_distance)
    return std::numeric_limits<float>::max();

  // Return the difference between the distances.
  // 返回距离间的差值
  const float model_distance = (model_point_1 - model_point_2).norm();
  return fabs(scene_distance - model_distance);
}

} // namespace segmatch
