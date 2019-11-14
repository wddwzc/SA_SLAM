#ifndef SEGMATCH_SEGMENTER_FACTORY_HPP_
#define SEGMATCH_SEGMENTER_FACTORY_HPP_

#include "segmatch/segmatch.hpp"
#include "segmatch/segmenters/segmenter.hpp"

namespace segmatch {

/// \brief Factory class for segmenters.
class SegmenterFactory {
 public:
  /// \brief Initializes a new instance of the SegmenterFactory class.
  /// \param params The current parameters of SegMatch.
  SegmenterFactory(SegMatchParams params);

  /// \brief Creates a segmenter.
  /// \returns Pointer to a new CorrespondencdRecognizer instance.
  // 创建分割器
  // 返回：一致性识别器实例的指针
  std::unique_ptr<Segmenter<MapPoint>> create() const;

 private:
  SegmenterParameters params_;
}; // class SegmenterFactory

} // namespace segmatch

#endif // SEGMATCH_SEGMENTER_FACTORY_HPP_
