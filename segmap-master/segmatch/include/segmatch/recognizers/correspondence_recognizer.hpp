#ifndef SEGMATCH_CORRESPONDENCE_RECOGNIZER_HPP_
#define SEGMATCH_CORRESPONDENCE_RECOGNIZER_HPP_

#include <vector>

#include "segmatch/common.hpp"
#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

/// \brief Base class for recognizing a model in a scene.
// 在场景中识别模型的虚基类
class CorrespondenceRecognizer {
 public:
  /// \brief Finalizes an instance of the CorrespondenceRecognizer class.
  virtual ~CorrespondenceRecognizer() = default;

  /// \brief Sets the current matches and tries to recognize the model.
  /// \param predicted_matches Vector of possible correspondences between model and scene.
  // 设置当前匹配并识别模型
  // 参数：模型与场景之间可能对应的向量
  virtual void recognize(const PairwiseMatches& predicted_matches) = 0;

  /// \brief Gets the candidate transformations between model and scene.
  /// \returns Vector containing the candidate transformations. Transformations are sorted in
  /// decreasing recognition quality order. If empty, the model was not recognized.
  // 获取模型和场景之间的候选变换关系
  // 参数：包含候选转换关系  该转换关系按照识别效果降序排列。如果是空的表明未识别
  virtual const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>&
  getCandidateTransformations() const = 0;

  /// \brief Gets the candidate clusters of matches between model and scene. Every cluster
  /// represents a possible recognition.
  /// \returns Vector containing the candidate clusters. Clusters are sorted in
  /// decreasing recognition quality order. If empty, the model was not recognized.
  // 获取模型和场景之间匹配的候选聚类，每一类表示一个可能的识别
  // 返回：由候选聚类组成，聚类按照识别效果降序排列，如果未空，则未识别
  virtual const std::vector<PairwiseMatches>& getCandidateClusters() const = 0;
}; // class CorrespondenceRecognizer

} // namespace segmatch

#endif // SEGMATCH_CORRESPONDENCE_RECOGNIZER_HPP_
