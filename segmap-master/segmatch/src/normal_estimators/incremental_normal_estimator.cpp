#include "segmatch/normal_estimators/incremental_normal_estimator.hpp"

#include <laser_slam/benchmarker.hpp>
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d.h>

namespace segmatch {

IncrementalNormalEstimator::IncrementalNormalEstimator(const float search_radius)
  : search_radius_(search_radius), kd_tree_(new pcl::search::KdTree<MapPoint>) {
}

void IncrementalNormalEstimator::notifyPointsTransformed(
      const kindr::minimal::QuatTransformationTemplate<float>& transformation) {
  BENCHMARK_BLOCK("SM.AddNewPoints.EstimateNormals.NotifyPointsTransformed");
  // Rotate the components of the covariance matrix by rewriting the the points as X := R*X + T
  // where R is the rotation matrix and T the translation vector of the transformation.

  // Get rotation and translation matrix.
  const EIGEN_ALIGN16 Eigen::Matrix3f R =  .getRotationMatrix();
  const EIGEN_ALIGN16 Eigen::Vector3f T = transformation.getPosition();
  const EIGEN_ALIGN16 Eigen::Matrix3f Rt = R.transpose();
  const EIGEN_ALIGN16 Eigen::RowVector3f Tt = T.transpose();

  for (size_t i = 0u; i < normals_.size(); ++i) {
    pcl::Vector3fMap normal = normals_[i].getNormalVector3fMap();
    Eigen::Matrix3f& sum_X_Xt = sum_X_Xt_[i];
    Eigen::Vector3f& sum_X = sum_X_[i];
    const float n = num_points_[i];

    // Transform the components of the covariance
	// X_ = R * X + T 带入
    sum_X_Xt = R * sum_X_Xt * Rt + R * sum_X * Tt + T * sum_X.transpose() * Rt + n * T * Tt;
    sum_X = n * T + R * sum_X;
    normal = transformation.getRotation().rotate(normal);
  }
}

void IncrementalNormalEstimator::clear() {
  sum_X_Xt_.clear();
  sum_X_.clear();
  num_points_.clear();
  normals_.clear();
}

// Auxiliary function for rearranging elements in a vector according to a mapping.
template <typename Container, typename Element>
void rearrangeElementsWithMapping(const std::vector<int>& mapping, const size_t new_size,
                                  const Element& default_value, Container& container) {
  Container new_container(new_size, default_value);
  for (size_t i = 0u; i < mapping.size(); ++i) {
    if (mapping[i] >= 0) new_container[mapping[i]] = std::move(container[i]);
  }
  container = std::move(new_container);
}

std::vector<bool> IncrementalNormalEstimator::updateNormals(
    const MapCloud& points, const std::vector<int>& points_mapping,
    const std::vector<int>& new_points_indices,
    PointsNeighborsProvider<MapPoint>& points_neighbors_provider) {

  // Rearrange the cached information according to the mapping.
  // 按照映射顺序重排序
  rearrangeElementsWithMapping(points_mapping, points.size(), Eigen::Matrix3f::Zero(), sum_X_Xt_);
  rearrangeElementsWithMapping(points_mapping, points.size(), Eigen::Vector3f::Zero(), sum_X_);
  rearrangeElementsWithMapping(points_mapping, points.size(), 0u, num_points_);
  rearrangeElementsWithMapping(points_mapping, points.size(), PclNormal(), normals_.points);

  // Scatter the contributions of the new points to the covariance matrices of each point's
  // neighborhood.
  // 将新点的贡献分散到每个点邻域的协方差矩阵中
  const std::vector<bool> is_normal_affected = scatterNormalContributions(
      points, new_points_indices, points_neighbors_provider);

  // Perform eigenvalues analysis on the affected point's covariances to determine their new
  // normals.
  recomputeNormals(points, is_normal_affected);
  return is_normal_affected;
}

std::vector<bool> IncrementalNormalEstimator::scatterNormalContributions(
    const MapCloud& points, const std::vector<int>& new_points_indices,
    PointsNeighborsProvider<MapPoint>& points_neighbors_provider) {
  BENCHMARK_BLOCK("SM.AddNewPoints.EstimateNormals.ScatterContributions");

	// 标记新加入的点
  std::vector<bool> is_new_point(points.size(), false);
  for (auto point_index : new_points_indices) is_new_point[point_index] = true;

  // Scatter information to all the points that are affected by the new points.
  // 将信息分散到所有受新点影响的点
  std::vector<bool> is_normal_affected(points.size(), false);
  // 遍历所有新点
  for (auto point_index : new_points_indices) {
    // Find neighbors.
	// 对新点进行邻域搜索
    std::vector<int> neighbors_indices = points_neighbors_provider.getNeighborsOf(point_index,
                                                                                  search_radius_);
    is_normal_affected[point_index] = true;

    const Eigen::Vector3f& source_point = points[point_index].getVector3fMap();
    for (const auto neighbor_index : neighbors_indices) {
      // Add contribution to the neighbor point.
	  // 将新点的贡献累计到邻域点上
	  // 标记受影响点
      is_normal_affected[neighbor_index] = true;
	  // 新点导致其中一个邻域点的协方差矩阵的变化量
      sum_X_Xt_[neighbor_index] += source_point * source_point.transpose();
      sum_X_[neighbor_index] += source_point;
	  // 邻域点数量
      ++num_points_[neighbor_index];

      // If the neighbor is an old point, then it also contributes to the normal of the new point.
	  // 如果邻域是旧点，也会对新点由贡献
	  // 将当前邻域点（旧点）的贡献累计到当前新点上
      if (!is_new_point[neighbor_index]) {
        const Eigen::Vector3f& neighbor_point = points[neighbor_index].getVector3fMap();
        sum_X_Xt_[point_index] += neighbor_point * neighbor_point.transpose();
        sum_X_[point_index] += neighbor_point;
        ++num_points_[point_index];
      }
    }
  }

  return is_normal_affected;
}

void IncrementalNormalEstimator::recomputeNormals(const MapCloud& points,
                                                  const std::vector<bool>& needs_recompute) {
  BENCHMARK_BLOCK("SM.AddNewPoints.EstimateNormals.UpdateNormals");

  CHECK(needs_recompute.size() == sum_X_Xt_.size());
  CHECK(needs_recompute.size() == sum_X_.size());
  CHECK(needs_recompute.size() == num_points_.size());
  CHECK(needs_recompute.size() == normals_.size());

  // Only recompute normals for affected points.
  size_t num_affected_normals = 0u;
  for (size_t i = 0u; i < normals_.size(); ++i) {
    if (needs_recompute[i]) {
      ++num_affected_normals;
      if (num_points_[i] >= 3u) {
        // If there are at least three points in the neighborhood, the normal vector is equal to
        // the eigenvector of the smallest eigenvalue.
		// 邻域点大于等于3个是，才会重新计算
        const float norm_factor = 1.0f / static_cast<float>(num_points_[i]);
		// C = 
        const EIGEN_ALIGN16 Eigen::Matrix3f covariance =
            (sum_X_Xt_[i] * norm_factor - sum_X_[i] * norm_factor * sum_X_[i].transpose() * norm_factor);
		// 求解协方差矩阵的特征向量和特征值，估计最小二乘平面法线和平面曲率
        pcl::solvePlaneParameters(covariance, normals_[i].normal_x, normals_[i].normal_y,
                                  normals_[i].normal_z, normals_[i].curvature);
		// 按照视点方向，将法向量对齐
        constexpr float view_point_component = std::numeric_limits<float>::max();
        pcl::flipNormalTowardsViewpoint(points[i], view_point_component, view_point_component,
                                        view_point_component, normals_[i].normal_x,
                                        normals_[i].normal_y, normals_[i].normal_z);
      } else {
        // Otherwise we don't have enough data to estimate the normal. Just set it to NaN.
		// 邻域点过少时，坐标置NaN
        normals_[i].normal_x = normals_[i].normal_y = normals_[i].normal_z = normals_[i].curvature
            = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  BENCHMARK_RECORD_VALUE("SM.AddNewPoints.EstimateNormals.AffectedNormals", num_affected_normals);
}

} // namespace segmatch
