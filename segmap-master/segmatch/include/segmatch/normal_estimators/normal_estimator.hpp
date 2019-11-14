#ifndef SEGMATCH_NORMAL_ESTIMATOR_HPP_
#define SEGMATCH_NORMAL_ESTIMATOR_HPP_

#include <laser_slam/common.hpp>

#include "segmatch/common.hpp"
#include "segmatch/points_neighbors_providers/points_neighbors_provider.hpp"

namespace segmatch {

/// \brief Base class for estimating normals in a point cloud.
class NormalEstimator {
 public:
  /// \brief Finalizes an instance of the NormalEstimator class.
  virtual ~NormalEstimator() = default;

  /// \brief Notifies the estimator that points have been transformed.
  /// \param transformation Linear transformation applied to the points.
  /// \remarks updateNormals() must still be called so that the normal vectors actually reflect the
  /// transformation.
  // 通知estimator点已经转换完成
  // 参数：对点进行的线性变换
  // 必须调用updateNormals()，以便法向量可以反映变换
  virtual void notifyPointsTransformed(
      const kindr::minimal::QuatTransformationTemplate<float>& transformation) = 0;

  /// \brief Clear all the normals and the associated information. Equivalent to removing all the
  /// points from the cloud.
  // 清空所有法线和相关信息，相当于清空所有点云
  virtual void clear() = 0;

  /// \brief Updates the normals of the points of a cloud.
  /// \param points The point cloud for which normals have to be computed.
  /// \param points_mapping Mapping from the indices of the points in the old point cloud to the
  /// indices of the points in the new point cloud since the last update.
  /// \param new_points_indices Indices of the points added to the point cloud since the
  /// last update.
  /// \param points_neighbors_provider Object for nearest neighbors searches.
  /// \returns Vector of booleans indicating which normals or curvatures changed after the update.
  // 更新点云的法线
  // 参数：需要计算法线的点云  自上一次更新以来，旧点云索引到新点云索引的映射  新加入的点云索引  最近邻索搜的对象
  // 返回：表示更新后法线或曲率变化的bool向量
  virtual std::vector<bool> updateNormals(
      const MapCloud& points, const std::vector<int>& points_mapping,
      const std::vector<int>& new_points_indices,
      PointsNeighborsProvider<MapPoint>& points_neighbors_provider) = 0;

  /// \brief Gets the current normal vectors of the point cloud.
  /// \returns Cloud containing the normal vectors.
  // 获取点云当前法向量
  // 返回：包含法向量的点云
  virtual const PointNormals& getNormals() const = 0;

  /// \brief Creates a normal estimator with the passed parameters.
  /// \param estimator_type Type of estimator. Can be "simple" or "incremental".
  /// \param radius_for_estimation_m The search radius used for the estimation of the normals.
  // 根据传入参数创建一个法向量估计器
  // 参数：估计器类型  搜索半径
  static std::unique_ptr<NormalEstimator> create(const std::string& estimator_type,
                                                 float radius_for_estimation_m);
}; // class NormalEstimator

} // namespace segmatch

#endif // SEGMATCH_NORMAL_ESTIMATOR_HPP_
