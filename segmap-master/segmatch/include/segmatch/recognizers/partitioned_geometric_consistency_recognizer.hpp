#ifndef SEGMATCH_PARTITIONED_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_
#define SEGMATCH_PARTITIONED_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_

#include "segmatch/parameters.hpp"
#include "segmatch/recognizers/graph_based_geometric_consistency_recognizer.hpp"

namespace segmatch {

/// \brief Recognizes a model in a scene using a graph-based approach. First a consistency graph
/// is constructed, where the nodes represent the matches and edges connect matches that are
/// pairwise consistent. Recognition finds a maximum clique matches that are pairwise consistent.
/// The partitioned approach assumes that the model is relatively small compared to the scene. This
/// allows to partition the space in a voxel grid with resolution equal to the size of the model
/// and efficiently discard all pairwise matches that are two or more voxels apart, i.e. too
/// distant to be consistent given the model size.
/// \remark The current implementation assumes that the model and the scene are almost planar,
/// and partitioning is necessary only on the X and Y axis. Extension to 3D is possible by using
/// 3D partitioning.
// 使用基于图的方法在场景中识别模型
// 先构建一致性图，节点表示匹配点，边连接成对关联的匹配点。识别就是找到成对关联匹配的最大团。
// 分割方法假定与场景相比模型较小，这使得能以模型的大小作为分辨率来划分体素网格中的空间
// 并有效丢掉分离两个或更多体素的成对匹配，即在给定模型尺寸下太远就不一致
// 现在的实现假定模型与场景近似一个平面，分割只在x y轴上是必要的，通过3D分割可能扩展到3D
class PartitionedGeometricConsistencyRecognizer : public GraphBasedGeometricConsistencyRecognizer {
 public:
  /// \brief Initializes a new instance of the PartitionedGeometricConsistencyRecognizer class.
  /// \param params The parameters of the geometry consistency grouping.
  /// \param max_model_radius Radius of the bounding cylinder of the model.
  // 初始化
  // 参数：参数  模型的边界圆柱半径
  PartitionedGeometricConsistencyRecognizer(const GeometricConsistencyParams& params,
                                            float max_model_radius) noexcept;

 protected:
  /// \brief Builds a consistency graph of the provided matches.
  /// \param predicted_matches Vector of possible correspondences between model and scene.
  /// \returns Graph encoding pairwise consistencies. Match \c predicted_matches[i] is represented
  /// by node \c i .
  // 根据提供匹配构建关联图
  // 参数：模型与场景的可能关联
  // 返回：图编码的成对关联
  ConsistencyGraph buildConsistencyGraph(const PairwiseMatches& predicted_matches) override;

 private:
  // Per-partition data
  // 预分割数据
  struct PartitionData { };

  // Find consistencies within a partition and add them to the consistency graph.
  // 找到一个分割中的关联，将它们加入关联图
  size_t findAndAddInPartitionConsistencies(const PairwiseMatches& predicted_matches,
                                          const std::vector<size_t>& partition_indices,
                                          ConsistencyGraph& consistency_graph) const;

  // Find consistencies between two partitions and add them to the consistency graph.
  // 找到两个分割间的关联，将它们加入关联图
  size_t findAndAddCrossPartitionConsistencies(const PairwiseMatches& predicted_matches,
                                             const std::vector<size_t>& partition_indices_1,
                                             const std::vector<size_t>& partition_indices_2,
                                             ConsistencyGraph& consistency_graph) const;

  float partition_size_;
}; // class PartitionedGeometricConsistencyRecognizer

} // namespace segmatch

#endif // SEGMATCH_PARTITIONED_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_
