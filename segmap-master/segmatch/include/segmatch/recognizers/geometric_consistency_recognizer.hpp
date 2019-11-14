#ifndef SEGMATCH_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_
#define SEGMATCH_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_

#include "segmatch/parameters.hpp"
#include "segmatch/recognizers/correspondence_recognizer.hpp"

namespace segmatch {

/// \brief Recognizes a model in a scene using the \c GeometricConsistencyRecognizer implementation
/// provided by the PCL. Recognition finds a subset of centroids in the model whose pairwise
/// distances are consistent with the matching centroids in the scene.
// 使用PCL提供的GeometricConsistencyRecognizer实现方式在场景中识别一个模型
// 识别找到匹配质心距离一致的分割块对子集
class GeometricConsistencyRecognizer : public CorrespondenceRecognizer {
 public:
  /// \brief Initializes a new instance of the GeometricConsistencyRecognizer class.
  /// \param params The parameters of the geometry consistency grouping.
  // noexcept 不抛出任何异常
  GeometricConsistencyRecognizer(const GeometricConsistencyParams& params) noexcept
    : params_(params) { }

  /// \brief Sets the current matches and tries to recognize the model.
  /// \param predicted_matches Vector of possible correspondences between model and scene.
  // 设置当前匹配，识别模型
  // 参数：模型和场景间的可能对应
  // override：如果派生类在虚函数声明时使用了override描述符，那么该函数必须重载其基类中的同名函数，否则代码将无法通过编译
  void recognize(const PairwiseMatches& predicted_matches) override;

  /// \brief Gets the candidate transformations between model and scene.
  /// \returns Vector containing the candidate transformations. Transformations are sorted in
  /// decreasing recognition quality order. If empty, the model was not recognized.
  // 获取模型与场景间的候选转换关系
  // 返回：候选转换关系，以匹配效果降序
  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>&
  getCandidateTransformations() const override {
    return candidate_transfomations_;
  }

  /// \brief Gets the candidate clusters of matches between model and scene. Every cluster
  /// represents a possible recognition.
  /// \returns Vector containing the candidate clusters. Clusters are sorted in
  /// decreasing recognition quality order. If empty, the model was not recognized.
  // 获取模型和场景间匹配的候选聚类
  // 返回：候选聚类
  const std::vector<PairwiseMatches>& getCandidateClusters() const override {
    return candidate_matches_;
  }

 private:
  // Candidate transformations and matches between model and scene.
  // 候选转换
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
  candidate_transfomations_;
  std::vector<PairwiseMatches> candidate_matches_;

  // The parameters of the geometry consistency grouping.
  GeometricConsistencyParams params_;
}; // class GeometricConsistencyRecognizer

} // namespace segmatch

#endif // SEGMATCH_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_
