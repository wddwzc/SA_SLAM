#include "segmatch/descriptors/descriptors.hpp"

#include <Eigen/Dense>
#include <glog/logging.h>
#include <laser_slam/common.hpp>

#include "segmatch/descriptors/cnn.hpp"
#include "segmatch/descriptors/eigenvalue_based.hpp"
#include "segmatch/descriptors/ensemble_shape_functions.hpp"

namespace segmatch {

// Descriptors的describe方法都是通过调用Descriptor的describe进而实现的

// 在Descriptor中，只有describe(Segment*)定义了，其它都是虚函数

Descriptor::Descriptor() {}

Descriptor::~Descriptor() {}

// 通过分割块和最新观测的特征来描述分割
// 参数：segment_ptr
void Descriptor::describe(Segment* segment_ptr) {
  describe(*segment_ptr, &segment_ptr->getLastView().features);
}

// Descriptors methods definition
Descriptors::Descriptors() {
  //TODO: Implement default construction from default parameters.
  CHECK(false) << "Construction of descriptors without parameters is not implemented.";
}

Descriptors::~Descriptors() {}

// 参数初始化 Descriptors
Descriptors::Descriptors(const DescriptorsParameters& parameters) {
  CHECK_GT(parameters.descriptor_types.size(), 0) << "Description impossible without a descriptor.";

  // Create the descriptors.
  for (size_t i = 0u; i < parameters.descriptor_types.size(); ++i) {
    if (parameters.descriptor_types[i] == "EigenvalueBased") {
	// 参数中默认使用的描述子类型
      descriptors_.push_back(std::unique_ptr<Descriptor>(
          new EigenvalueBasedDescriptor(parameters)));
    } else if (parameters.descriptor_types[i] == "EnsembleShapeFunctions") {
      descriptors_.push_back(std::unique_ptr<Descriptor>(new EnsembleShapeFunctions(parameters)));
    } else if (parameters.descriptor_types[i] == "CNN") {
	// 使用CNN时使用的描述子类型
      descriptors_.push_back(std::unique_ptr<Descriptor>(new CNNDescriptor(parameters)));
    } else {
      CHECK(false) << "The descriptor '" << parameters.descriptor_types[i] <<
          "' was not implemented.";
    }
  }
}

// 通过分割块和最新观测描述，该方法被子类调用
// 参数：segment_ptr
void Descriptors::describe(Segment* segment_ptr) {
  describe(*segment_ptr, &segment_ptr->getLastView().features);
}

// 参数：segment  features_ptr
void Descriptors::describe(const Segment& segment, Features* features) {
  CHECK_NOTNULL(features)->clear();
  CHECK_GT(descriptors_.size(), 0) << "Description impossible without a descriptor.";
  for (size_t i = 0u; i < descriptors_.size(); ++i) {
    descriptors_[i]->describe(segment, features);
  }
}

// 参数：Segmented_Cloud::Ptr
void Descriptors::describe(SegmentedCloud* segmented_cloud_ptr) {
  CHECK_NOTNULL(segmented_cloud_ptr);
  CHECK_GT(descriptors_.size(), 0) << "Description impossible without a descriptor.";
  for (size_t i = 0u; i < descriptors_.size(); ++i) {
    descriptors_[i]->describe(segmented_cloud_ptr);
  }
}

// 返回描述子总维度
unsigned int Descriptors::dimension() const {
  CHECK_GT(descriptors_.size(), 0) << "Description impossible without a descriptor.";
  unsigned int dimension = 0;
  for (size_t i = 0u; i < descriptors_.size(); ++i) {
    dimension += descriptors_[i]->dimension();
  }
  return dimension;
}

} // namespace segmatch
