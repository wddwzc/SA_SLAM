#ifndef SEGMATCH_SEGMATCH_HPP_
#define SEGMATCH_SEGMATCH_HPP_

#include <cmath>
#include <queue>
#include <string>

#include <laser_slam/common.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "segmatch/common.hpp"
#include "segmatch/descriptors/descriptors.hpp"
#include "segmatch/database.hpp"
#include "segmatch/local_map.hpp"
#include "segmatch/opencv_random_forest.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/recognizers/correspondence_recognizer.hpp"
#include "segmatch/segmented_cloud.hpp"
#include "segmatch/segmenters/segmenter.hpp"

namespace segmatch {

struct SegMatchParams {
  double segmentation_radius_m;
  double segmentation_height_above_m;
  double segmentation_height_below_m;

  bool filter_boundary_segments;
  double boundary_radius_m;
  bool filter_duplicate_segments;
  double centroid_distance_threshold_m;
  laser_slam::Time min_time_between_segment_for_matches_ns;
  bool check_pose_lies_below_segments = false;

  /// \brief The method used for estimating the point normals.
  std::string normal_estimator_type;
  /// \brief Radius of the neighborhood considered for the estimation of the point normals.
  float radius_for_normal_estimation_m;

  LocalMapParameters local_map_params;
  DescriptorsParameters descriptors_params;
  SegmenterParameters segmenter_params;
  ClassifierParams classifier_params;
  GeometricConsistencyParams geometric_consistency_params;
};

class SegMatch {
 public:
  /// \brief Type of the local map.
  // 局部地图的类型
  typedef LocalMap<PclPoint, MapPoint> LocalMapT;

  explicit SegMatch(const SegMatchParams& params);
  SegMatch();
  ~SegMatch();

  /// \brief Initialize SegMatch.
  void init(const SegMatchParams& params,
            unsigned int num_tracks = 1u);

  /// \brief Convenience function for setting part of the params.
  // 设置部分参数
  void setParams(const SegMatchParams& params);

  /// \brief Process a source cloud.
  // 处理点云并设置为源点云
  void processAndSetAsSourceCloud(LocalMapT& local_map,
                                  const laser_slam::Pose& latest_pose,
                                  unsigned int track_id = 0u);

  /// \brief Process a target cloud.
  // 处理点云并设置为目标点云
  void processAndSetAsTargetCloud(MapCloud& target_cloud);

  /// \brief Transfer the source cloud to the target cloud.
  // 将源点云转为目标点云
  void transferSourceToTarget(unsigned int track_id = 0u,
                              laser_slam::Time timestamp_ns = 0u);

  /// \brief Find matches between the source and the target clouds.
  // 找到源点云与目标点云间的匹配
  PairwiseMatches findMatches(PairwiseMatches* matches_after_first_stage = NULL,
                              unsigned int track_id = 0u,
                              laser_slam::Time timestamp_ns = 0u);

  /// \brief Filter the matches, removing the ones whose segments are too close in time and thus
  /// are likely to be self-matches.
  /// \param predicted_matches The predicted matches that need to be filtered.
  /// \param track_id ID of the robot for which matches are processed.
  /// \returns The filtered matches.
  // 筛选匹配，移除时间太近所以更像自身匹配的匹配对
  // 参数：需要筛选的预测匹配  处理匹配的机器人ID
  // 返回：筛选后的匹配
  PairwiseMatches filterMatches(const PairwiseMatches& predicted_matches, unsigned int track_id);

  /// \brief Try to recognize the local map in the target map.
  /// \param predicted_matches The predicted matches between segments in the local cloud and
  /// segments in the target map.
  /// \param track_id ID of the robot for which matches are processed.
  /// \param timestamp_ns Timestamp of the current pose.
  /// \param loop_closure Pointer to a structure where the details of the loop closure are
  /// returned.
  /// \returns The matches accepted by the recognition. If empty, the local map couldn't be
  /// recognized.
  // 识别目标地图中的局部地图
  // 参数：局部地图中的分割和目标地图中的分割的预测匹配  处理匹配的机器人ID  当前位置的时间戳  闭环详细信息
  // 返回：被识别的匹配项
  const PairwiseMatches& recognize(const PairwiseMatches& predicted_matches, unsigned int track_id,
                                   laser_slam::Time timestamp_ns = 0u,
                                   laser_slam::RelativePose* loop_closure = nullptr);

  void update(const std::vector<laser_slam::Trajectory>& trajectories);

  /// \brief Get the internal representation of the source cloud.
  // 获取源点云的内部表示形式
  void getSourceRepresentation(PointICloud* source_representation,
                               const double& distance_to_raise = 0.0,
                               unsigned int track_id = 0u) const;

  /// \brief Get the internal representation of the target cloud.
  // 获取目标点云的内部表示形式
  void getTargetRepresentation(PointICloud* target_representation,
                               bool get_compressed = false) const;
	
  // 获取源点云重构
  void getSourceReconstruction(PointICloud* source_reconstruction,
                               unsigned int track_id = 0u) const;
  
  // 获取目标点云重构
  void getTargetReconstruction(PointICloud* target_reconstruction,
                               bool get_compressed = false) const;

  // 获取目标分割质心
  void getTargetSegmentsCentroids(PointICloud* segments_centroids) const;

  // 获取源点云语义
  void getSourceSemantics(PointICloud* source_semantics,
                          const double& distance_to_raise = 0.0,
                          unsigned int track_id = 0u) const;

  // 获取目标分割质心（轨迹ID作为intensity值）
  void getTargetSegmentsCentroidsWithTrajIdAsIntensity(PointICloud* segments_centroids) const;

  // 获取源点云分割质心
  void getSourceSegmentsCentroids(PointICloud* segments_centroids,
                                  unsigned int track_id = 0u) const;

  // 获取源点云作为已分割点云
  SegmentedCloud getSourceAsSegmentedCloud(unsigned int track_id = 0u) const {
    if (segmented_source_clouds_.find(track_id) != segmented_source_clouds_.end()) {
      return segmented_source_clouds_.at(track_id);
    } else {
      return SegmentedCloud();
    }
  };

  SegmentedCloud getTargetAsSegmentedCloud() const { return segmented_target_cloud_; };

  // 获取过去匹配的表现形式
  void getPastMatchesRepresentation(PointPairs* past_matches,
                                    PointPairs* invalid_past_matches = NULL) const;

  // 获取最新匹配
  void getLatestMatch(int64_t* time_a, int64_t* time_b,
                      Eigen::Matrix4f* transform_a_b,
                      std::vector<int64_t>* collector_times) const;

  /// \brief Process a cloud and return the segmented cloud.
  // 处理点云，返回分割后的
  void processCloud(MapCloud& cloud, SegmentedCloud* segmented_cloud);

  /// \brief Get the descriptors dimension.
  // 获取描述子维度
  unsigned int getDescriptorsDimension() const { return descriptors_->dimension(); };

  /// \brief Train the classifier.
  // 训练分类器
  void trainClassifier(const Eigen::MatrixXd& features, const Eigen::MatrixXd& labels) {
    classifier_->train(features, labels);
  };

  /// \brief Test the classifier.
  // 测试分类器
  void testClassifier(const Eigen::MatrixXd& features, const Eigen::MatrixXd& labels,
                      Eigen::MatrixXd* probabilities = NULL) {
    classifier_->test(features, labels, probabilities);
  };

  /// \brief Save the classifier.
  // 保存分类器
  void saveClassifier(const std::string& filename) { classifier_->save(filename); };

  /// \brief Load a classifier.
  // 加载分类器
  void loadClassifier(const std::string& filename) { classifier_->load(filename); };

  // 计算特征距离
  void computeFeaturesDistance(const Eigen::MatrixXd& f1, const Eigen::MatrixXd& f2,
                               Eigen::MatrixXd* f_out) const {
    classifier_->computeFeaturesDistance(f1, f2, f_out);
  };

  // 获取分割块的位置
  void getSegmentationPoses(std::vector<laser_slam::Trajectory>* poses) const {
    CHECK_NOTNULL(poses);
    *poses = segmentation_poses_;
  };

  segmatch::PairwiseMatches  getFilteredMatches() const { return last_filtered_matches_; };
  segmatch::PairwiseMatches  getPredictedMatches() const { return last_predicted_matches_; };

  void getLoopClosures(std::vector<laser_slam::RelativePose>* loop_closures) const;

  void alignTargetMap();

  void displayTimings() const;

  void saveTimings() const;

  void exportDescriptorsData() const { descriptors_->exportData(); };

  std::vector<database::MergeEvent> getMergeEvents() { return merge_events_; }

 private:
  laser_slam::Time findTimeOfClosestSegmentationPose(const segmatch::Segment& segment) const;

  void filterNearestSegmentsInCloud(SegmentedCloud& cloud, double minimum_distance_m,
                                    unsigned int n_nearest_segments = 2u);

  SegMatchParams params_;

  std::unique_ptr<Segmenter<MapPoint>> segmenter_;
  std::unique_ptr<Descriptors> descriptors_;
  // Create one recognizer per track, since the incremental recognizer caches track-specific data.
  // 每条轨迹创建一个识别器，因为增量识别器缓存特定轨迹的数据
  std::vector<std::unique_ptr<CorrespondenceRecognizer>> recognizers_;

  //TODO(Renaud or Daniel): rename.
  std::unique_ptr<OpenCvRandomForest> classifier_;

  // worker的id与segmented_cloud的映射关系
  std::unordered_map<unsigned int, SegmentedCloud> segmented_source_clouds_;
  // Segments that have been renamed in the last segmentation step (e.g. because of merging). Each
  // pair contains the original and new IDs of the renamed segments.
  // 最近一次分割步骤重命名的分割块，每个pair包含原始和新的ID
  std::unordered_map<unsigned int, std::vector<std::pair<Id, Id>>> renamed_segments_;
  unsigned int last_processed_source_cloud_ = 0u;

  // 定位时使用的目标点云分割
  SegmentedCloud segmented_target_cloud_;
  std::vector<SegmentedCloud> target_queue_;

  // Contains the poses where segmentation and matching was performed.
  // 包含执行分割和匹配的位置
  std::vector<laser_slam::Trajectory> segmentation_poses_;

  PairwiseMatches last_filtered_matches_;
  PairwiseMatches last_predicted_matches_;

  std::vector<laser_slam::RelativePose> loop_closures_;

  Eigen::Matrix4f last_transformation_;

  std::vector<double> loops_timestamps_;

  std::vector<database::MergeEvent> merge_events_;

  // Filtering parameters.
  // 滤波参数
  static constexpr double kCylinderHeight_m = 40;
  static constexpr unsigned int kMaxNumberOfCloudToTransfer = 1u;

  static constexpr laser_slam::Time kMaxTimeDiffBetweenSegmentAndPose_ns = 20000000000u;

}; // class SegMatch

} // namespace segmatch

#endif // SEGMATCH_SEGMATCH_HPP_
