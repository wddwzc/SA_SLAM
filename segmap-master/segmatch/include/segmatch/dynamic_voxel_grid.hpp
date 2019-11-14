#ifndef SEGMATCH_DYNAMIC_VOXEL_GRID_HPP_
#define SEGMATCH_DYNAMIC_VOXEL_GRID_HPP_

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include <glog/logging.h>
#include <kindr/minimal/quat-transformation.h>
#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

namespace segmatch {

/// \brief A grid of cubic volume cells.
///
/// Points inserted in the grid are assigned to their respective voxels. The grid provides a
/// downsampled view of the points and supports removing of voxels according to a predicate.
/// The grid distinguishes between <em>active voxels</em> (voxels that contain a minimum number of
/// points) and \e inactive voxels.
/// \remark The class is \e not thread-safe. Concurrent access to the class results in undefined
/// behavior.

// 一个基于立方体单元的栅格
// 点插入栅格中，会关联到各自的体素。栅格可以进行降采样或移除体素。栅格区分激活体素和非激活体素
// 类型不是线程安全的，并发地访问该类会导致不确定行为

template<
  typename InputPointT,
  typename VoxelPointT,
  typename IndexT = uint64_t,
  uint8_t bits_x = 20,
  uint8_t bits_y = 20,
  uint8_t bits_z = 20>
class DynamicVoxelGrid {
 public:
  typedef typename pcl::PointCloud<InputPointT> InputCloud;
  typedef typename pcl::PointCloud<VoxelPointT> VoxelCloud;

	// 静态断言即，在编译时期就能发现问题所在，而不用等到运行时期（因为有些时候并不是所有断言，一次就能运行出来）
	// is_integral 检测IndexT是否是整型
  static_assert(std::is_integral<IndexT>::value && std::is_unsigned<IndexT>::value,
                "IndexT must be an unsigned integral type");
  static_assert(bits_x + bits_y + bits_z <= sizeof(IndexT) * 8,
                "The number of bits required per dimension is bigger than the size of IndexT");
  static_assert(bits_x > 0 && bits_y > 0 && bits_z > 0,
                "The index requires at least one bit per dimension");
	// 检测 point type 是否包含 xyz
  static_assert(pcl::traits::has_xyz<InputPointT>::value,
                "InputPointT must be a structure containing XYZ coordinates");
  static_assert(pcl::traits::has_xyz<VoxelPointT>::value,
                "VoxelPointT must be a structure containing XYZ coordinates");

  /// \brief Initializes a new instance of the DynamicVoxelGrid class.
  /// \param resolution Edge length of the voxels.
  /// \param min_points_per_voxel Minimum number of points that a voxel must contain in order to be
  /// considered \e active.
  /// \param origin The point around which the grid is centered.
  // 初始化：
  // resolution 体素边缘长度
  // min_point_per_voxel 体素作为激活态必须包含的最小点数
  // origin 网格的中心点
  
  // 体素栅格参数如下：
  // 分辨率 resolution 0.1
  // 边长 n_voxels_x 1^20=1048576
  // 栅格尺寸 grid_size_ 104857.6
  // 原点偏移应该为默认初始化的 (0,0,0)
  // min_corner_和max_corner_就是边长折半作偏移
  // indexing_offset_原点坐标偏移
  DynamicVoxelGrid(const float resolution, const int min_points_per_voxel,
                   const InputPointT& origin = InputPointT())
	// 单元长度
    : resolution_(resolution)
	// 激活体素最小数量
    , min_points_per_voxel_(min_points_per_voxel)
	// 栅格原点
    , origin_(origin)
	// 栅格的尺寸
    , grid_size_(
        resolution * static_cast<float>(n_voxels_x),
        resolution * static_cast<float>(n_voxels_y),
        resolution * static_cast<float>(n_voxels_z))
	// 原点偏移
    , origin_offset_(origin.getVector3fMap())
	// 索引偏移
    , indexing_offset_(grid_size_ / 2.0f - origin_offset_)
	// 实际世界坐标到索引坐标的放缩比例
    , world_to_grid_(1.0f / resolution)
	// 坐标最小点
    , min_corner_(origin_offset_ - grid_size_ / 2.0f)
	// 坐标最大点
    , max_corner_(origin_offset_ + grid_size_ / 2.0f)
	// 激活体素
    , active_centroids_(new VoxelCloud())
	// 非激活体素
    , inactive_centroids_(new VoxelCloud())
	// 位置转换
    , pose_transformation_()
	// 索引转换
    , indexing_transformation_() {

    // Validate inputs.
    CHECK_GT(resolution, 0.0f);
    CHECK_GE(min_points_per_voxel, 1);
  }

  /// \brief Move constructor for the DynamicVoxelGrid class.
  /// \param other Object that has to be moved into the new instance.
  // 拷贝构造
  // std::move() 左值转成右值
  DynamicVoxelGrid(DynamicVoxelGrid&& other)
    : resolution_(other.resolution_)
    , min_points_per_voxel_(other.min_points_per_voxel_)
    , origin_(std::move(other.origin_))
    , grid_size_(std::move(other.grid_size_))
    , origin_offset_(std::move(other.origin_offset_))
    , indexing_offset_(std::move(other.indexing_offset_))
    , world_to_grid_(other.world_to_grid_)
    , min_corner_(std::move(other.min_corner_))
    , max_corner_(std::move(other.max_corner_))
    , active_centroids_(std::move(other.active_centroids_))
    , inactive_centroids_(std::move(other.inactive_centroids_))
    , voxels_(std::move(other.voxels_))
    , pose_transformation_(std::move(other.pose_transformation_))
    , indexing_transformation_(std::move(other.indexing_transformation_)) {
  }

  /// \brief Inserts a point cloud in the voxel grid.
  /// Inserting new points updates the X, Y, Z coordinates of the points, but leaves any extra
  /// fields untouched.
  /// \remark Insertion invalidates any reference to the centroids.
  /// \param new_cloud The new points that must be inserted in the grid.
  /// \returns Indices of the centroids of the voxels that have become \e active after the
  /// insertion.
  // 在体素栅格中插入点云（更新xyz，其它额外字段不变）
  // 参数：要插入的点云
  // 返回：插入后，变成激活态的体素质心的索引
  std::vector<int> insert(const InputCloud& new_cloud);

  /// \brief Result of a removal operation.
  /// \remark Enabled only for predicates of the form: <tt>bool p(const VoxelPointT&)</tt>
  // 移除操作的结果
  // is_convertible<A, B> A是否可以隐式转换到B
  // function<T> 将T转换为function对象
  // enable_if<bool Cond, class T> 如果Cond为true，type=T，否则无法通过编译
  // 此处类型为 vector<bool>
  template<typename Func>
  using RemovalResult = typename std::enable_if<
    std::is_convertible<Func, std::function<bool(const VoxelPointT&)>>::value,
    std::vector<bool>>::type;

  /// \brief Removes from the grid a set of voxels satisfying the given predicate.
  /// \remark Removal invalidates any reference to the centroids.
  /// \returns Vector indicating, for each active voxel index, if the centroid has been removed or
  /// not.
  // 从网格中删除满足给定谓词的一组体素（会使引用无效）
  // vector标记激活体素的质心是否被移除
  template <typename Func>
  RemovalResult<Func> removeIf(Func predicate);

  /// \brief Compute the index of the voxel containing the specified point.
  // 计算得到包含指定点的体素索引
  template<typename PointXYZ_>
  IndexT getIndexOf(const PointXYZ_& point) const;

  /// \brief Apply a pose transformation to the voxel grid.
  /// \remark Multiple transformations are cumulative.
  /// \param transformation The transformation to be applied to the grid.
  // 对体素栅格进行位姿变换（多次变换累计）
  // 参数：网格要进行的变换
  void transform(const kindr::minimal::QuatTransformationTemplate<float>& transformation);

  /// \brief Clears the dynamic voxel grid, removing all the points it contains and resetting the
  /// transformations.
  // 清除动态体素栅格，移除所有点并重置变换
  void clear();

  /// \brief Returns a reference to the centroids of the active voxels.
  /// \remark Modifying the X, Y, Z components of the points in the returned cloud results in
  /// undefined behavior.
  /// \returns The centroids of the active voxels.
  // 返回激活体素质心的引用
  // 返回：激活体素的质心
  inline VoxelCloud& getActiveCentroids() const { return *active_centroids_; }

  /// \brief Returns a reference to the centroids of the inactive voxels.
  /// \remark Modifying the X, Y, Z components of the points in the returned cloud results in
  /// undefined behavior.
  /// \returns The centroids of the inactive voxels.
  // 返回非激活体素质心的引用
  // 返回：非激活体素的质心
  inline VoxelCloud& getInactiveCentroids() const { return *inactive_centroids_; }

  /// \brief Dump informations about the voxels contained in the grid.
  // 转储栅格中的体素信息
  void dumpVoxels() const;

 private:
  // A point with its voxel index.
  // 包含体素索引的点
  struct IndexedPoint_ {
    IndexedPoint_(const InputPointT& point, const IndexT& voxel_index)
      : point(point), voxel_index(voxel_index) {
    }

    InputPointT point;
    IndexT voxel_index;
  };
  typedef std::vector<IndexedPoint_> IndexedPoints_;

  // A voxel in the grid.
  // 栅格中的体素结构
  struct Voxel_ {
    Voxel_()
      : centroid(nullptr), index(0), num_points(0) {
    }

    Voxel_(VoxelPointT* centroid, const IndexT& index, const uint32_t num_points)
      : centroid(centroid), index(index), num_points(num_points) {
    }

    Voxel_(const Voxel_& other)
      : centroid(other.centroid), index(other.index), num_points(other.num_points) {
    }

    VoxelPointT* centroid;
    IndexT index;
    uint32_t num_points;
  };

  // The data necessary to construct a voxel.
  // 构建体素的必要数据
  struct VoxelData_ {
    Voxel_* old_voxel;
    typename IndexedPoints_::iterator points_begin;
    typename IndexedPoints_::iterator points_end;
  };

  // Compute the voxel indices of a point cloud and sort the points in increasing voxel index
  // order.
  // 计算点云的体素索引并以索引升序排列
  IndexedPoints_ indexAndSortPoints_(const InputCloud& points) const;

  // Create a voxel staring from the data about the point it contains and insert it in the voxels
  // and centroids vectors. Returns true if the new points inserted triggered the voxel.
  // 根据包含的数据创建体素，并将其加入体素和质心序列
  // 如果插入时触发了体素返回true
  bool createVoxel_(const IndexT index, const VoxelData_& data, std::vector<Voxel_>& new_voxels,
                    VoxelCloud& new_active_centroids, VoxelCloud& new_inactive_centroids);

  // Removes the centroids at the specified pointers. The pointers must be sorted in increasing
  // order.
  // 移除特定指针上的质心，指针必须升序排列
  std::vector<bool> removeCentroids_(VoxelCloud& target_cloud, std::vector<VoxelPointT*> to_remove);

  // The centroids of the voxels containing enough points.
  // 包含足够点的体素质心
  std::unique_ptr<VoxelCloud> active_centroids_;
  std::unique_ptr<VoxelCloud> inactive_centroids_;

  // The voxels in the point cloud.
  // 体素序列
  // 参数：指向active_centroids_或inactive_centroids_对应质心的指针  体素索引  点的数量
  std::vector<Voxel_> voxels_;

  // Properties of the grid.
  // 栅格属性
  const float resolution_;
  const int min_points_per_voxel_;
  const InputPointT origin_;

  // Size of the voxel grid.
  // 体素栅格尺寸
  static constexpr IndexT n_voxels_x = (IndexT(1) << bits_x);
  static constexpr IndexT n_voxels_y = (IndexT(1) << bits_y);
  static constexpr IndexT n_voxels_z = (IndexT(1) << bits_z);

  // Variables needed for conversion from world coordinates to voxel index.
  // 将世界坐标转换为体素索引所需的变量
  const Eigen::Vector3f grid_size_;
  const Eigen::Vector3f origin_offset_;
  const Eigen::Vector3f indexing_offset_;
  const Eigen::Vector3f min_corner_;
  const Eigen::Vector3f max_corner_;
  float world_to_grid_;
  kindr::minimal::QuatTransformationTemplate<float> pose_transformation_;
  kindr::minimal::QuatTransformationTemplate<float> indexing_transformation_;
}; // class DynamicVoxelGrid

// Short name macros for Dynamic Voxel Grid (DVG) template declaration and
// specification.
#define _DVG_TEMPLATE_DECL_ typename InputPointT, typename VoxelPointT, typename IndexT, uint8_t \
  bits_x, uint8_t bits_y, uint8_t bits_z
#define _DVG_TEMPLATE_SPEC_ InputPointT, VoxelPointT, IndexT, bits_x, bits_y, bits_z

//=================================================================================================
//    DynamicVoxelGrid public methods implementation
//=================================================================================================

// 根据规则函数Func predicate，删除当前体素栅格中符合要求的点
// voxels_、active_centroids_和inactive_centroids_分别操作
// 返回：激活体素中的移除标志序列

// 该函数声明相当于 std::vector<bool> removeIf(Func predicate)
template<_DVG_TEMPLATE_DECL_>
template <typename Func>
inline DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::RemovalResult<Func>
DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::removeIf(Func predicate) {
  // Setup iterators
  auto v_read = voxels_.begin();
  const auto v_end = voxels_.end();

  // Returns a reference to the point cloud containing the centroid of the
  // specified voxel.
  // 返回包含特定体素的质心的容器
  // get_centroids_container_for 作为函数指针
  std::vector<VoxelPointT*> active_centroids_to_remove;
  std::vector<VoxelPointT*> inactive_centroids_to_remove;
  // 判断输入体素是否为激活太，返回相应的remove容器
  auto get_centroids_container_for = [&](const Voxel_& voxel)
      -> std::vector<VoxelPointT*>& {
    if (voxel.num_points >= min_points_per_voxel_) {
      return active_centroids_to_remove;
    } else {
      return inactive_centroids_to_remove;
    }
  };

  // Remove the voxels and collect the pointers of the centroids that must be
  // removed.
  // 移除体素并收集必须移除质心的指针
  // 循环终止条件（尾后 或 当前质心不需要清除）
  while(v_read != v_end && !predicate(*(v_read->centroid)))
    ++v_read;

  // 说明都不需要删除，返回全是false的序列
  if (v_read == v_end)
    return std::vector<bool>(active_centroids_->size(), false);
  
  // 将第一个需要remove的加入对应序列
  auto v_write = v_read;
  get_centroids_container_for(*v_read).push_back(v_read->centroid);
  ++v_read;

  // 将半径范围内的点向前填充，需要删除的加入相应队列
  for(; v_read != v_end; ++v_read) {
	// 判断是否在50米范围内f
    if(!predicate(*(v_read->centroid))) {
      // Copy the centroid, updating the pointer from the voxel.
      *v_write = *v_read;
      v_write->centroid -= get_centroids_container_for(*v_read).size();
      ++v_write;
    } else {
      // Keep track of the voxels that need to be deleted.
      get_centroids_container_for(*v_read).push_back(v_read->centroid);
    }
  }

  // 直接删除末尾的元素，剩下的就是需要保留的
  voxels_.erase(v_write, v_end);

  // Actually remove centroids
  // 根据以上得到的需要移除的质心序列，移除
  removeCentroids_(*inactive_centroids_, inactive_centroids_to_remove);
  return removeCentroids_(*active_centroids_, active_centroids_to_remove);
}

} // namespace segmatch

#endif // SEGMATCH_DYNAMIC_VOXEL_GRID_HPP_
