#include "segmatch/segmenters/segmenter_factory.hpp"

#include "segmatch/segmenters/euclidean_segmenter.hpp"
#include "segmatch/segmenters/incremental_segmenter.hpp"
#include "segmatch/segmenters/smoothness_constraints_segmenter.hpp"

namespace segmatch {

SegmenterFactory::SegmenterFactory(SegMatchParams params)
  : params_(params.segmenter_params) {
}

// 参数中采用的分割器类型均为IncrementalEuclideanDistance
std::unique_ptr<Segmenter<MapPoint>> SegmenterFactory::create() const {
  if (params_.segmenter_type == "SimpleSmoothnessConstraints") {
    return std::unique_ptr<Segmenter<MapPoint>>(
        new SmoothnessConstraintsSegmenter<MapPoint>(params_));
  } else if (params_.segmenter_type == "IncrementalEuclideanDistance") {
  // 目前参数中均选取该分割器
    return std::unique_ptr<Segmenter<MapPoint>>(
        new IncrementalSegmenter<MapPoint, EuclideanDistance>(params_));
  } else if (params_.segmenter_type == "IncrementalSmoothnessConstraints") {
    return std::unique_ptr<Segmenter<MapPoint>>(
        new IncrementalSegmenter<MapPoint, SmoothnessConstraints>(params_));
  } else if (params_.segmenter_type == "SimpleEuclideanDistance") {
    return std::unique_ptr<Segmenter<MapPoint>>(new EuclideanSegmenter<MapPoint>(params_));
  } else {
    LOG(FATAL) << "Invalid segmenter type specified: " << params_.segmenter_type;
    throw std::invalid_argument("Invalid segmenter type specified: " + params_.segmenter_type);
  }
}

} // namespace segmatch
