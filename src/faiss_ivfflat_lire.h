#pragma once

#include <faiss/IndexIVFFlat.h>

#include <cstdint>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace baseline_eval {

struct LireConfig {
  uint32_t initial_nlist{0};
  uint32_t max_nlist{0};
  uint32_t split_threshold{0};
  uint32_t reassign_k{64};
  uint32_t seed{1337};
  uint32_t max_splits_per_batch{100000};
  uint32_t maintenance_threads{1};
  uint32_t insert_threads{1};
  bool async_maintenance{false};
  bool enable_reassign{true};
  bool validate_invariants{false};
};

struct LireMaintenanceEvent {
  uint64_t job_id{0};
  faiss::idx_t triggering_list{-1};
  double request_ms{0.0};
  double start_ms{0.0};
  double compute_done_ms{0.0};
  double commit_done_ms{0.0};
  double end_ms{0.0};
  uint32_t affected_lists{0};
  std::vector<faiss::idx_t> affected_list_ids;
  uint32_t split_count{0};
  uint64_t reassigned_vectors{0};
};

class IndexIVFFlatLIRE final : public faiss::IndexIVFFlat {
 public:
  IndexIVFFlatLIRE(int d, const LireConfig& config);
  ~IndexIVFFlatLIRE() override;

  void train(faiss::idx_t n, const float* x) override;
  void add_with_ids(faiss::idx_t n, const float* x, const faiss::idx_t* xids) override;
  void search(faiss::idx_t n,
              const float* x,
              faiss::idx_t k,
              float* distances,
              faiss::idx_t* labels,
              const faiss::SearchParameters* params = nullptr) const override;
  void reset() override;
  size_t remove_ids(const faiss::IDSelector& sel) override;

  void start_maintenance_workers();
  void wait_for_maintenance();
  void stop_maintenance_workers();
  void reset_maintenance_timeline();
  faiss::idx_t published_rows() const;
  std::vector<LireMaintenanceEvent> maintenance_events() const;
  void search_at_frontier(faiss::idx_t n,
                          const float* x,
                          faiss::idx_t k,
                          float* distances,
                          faiss::idx_t* labels,
                          faiss::idx_t visible_rows) const;

  uint32_t active_nlist() const;
  uint64_t split_count() const;
  uint64_t reassign_scanned() const;
  uint64_t reassign_moved() const;

 private:
  class ActiveCentroidIndex;

  struct ListData {
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;

    size_t size() const { return ids.size(); }
    const float* row(size_t i, int d) const {
      return vectors.data() + i * static_cast<size_t>(d);
    }
  };

  struct Move {
    faiss::idx_t id{-1};
    faiss::idx_t source{-1};
    faiss::idx_t target{-1};
    std::vector<float> vector;
  };

  struct MaintenanceResult {
    uint32_t split_count{0};
    uint64_t reassigned_vectors{0};
    std::vector<faiss::idx_t> affected_lists;
  };

  ActiveCentroidIndex* active_quantizer();
  const ActiveCentroidIndex* active_quantizer() const;

  float l2(const float* a, const float* b) const;
  const float* centroid(faiss::idx_t slot) const;
  float* centroid(faiss::idx_t slot);

  ListData read_list(faiss::idx_t slot) const;
  void write_list(faiss::idx_t slot, const ListData& list);
  void rebuild_quantizer();
  faiss::idx_t nearest_active(const float* vector) const;
  std::vector<faiss::idx_t> nearest_centroid_slots(
      const float* center,
      faiss::idx_t excluded,
      uint32_t k) const;

  bool split_balanced_kmeans(
      const ListData& input,
      std::vector<uint8_t>* labels,
      std::vector<float>* centers);
  std::vector<faiss::idx_t> split_one(faiss::idx_t slot,
                                      bool* did_split,
                                      uint64_t* reassigned_vectors);
  std::vector<faiss::idx_t> collect_and_reassign(
      faiss::idx_t old_slot,
      const std::vector<float>& old_center,
      faiss::idx_t child0,
      faiss::idx_t child1,
      const std::vector<faiss::idx_t>& neighbors,
      uint64_t* reassigned_vectors);
  uint64_t apply_moves(const std::vector<Move>& moves);
  MaintenanceResult run_maintenance(const std::vector<faiss::idx_t>& touched);
  void enqueue_maintenance(faiss::idx_t slot);
  void maintenance_worker();
  void validate_state() const;
  std::shared_mutex& list_mutex(faiss::idx_t slot) const;

  LireConfig config_;
  uint32_t initial_nlist_{0};
  uint32_t max_nlist_{0};
  uint32_t active_nlist_{0};
  std::vector<float> centroids_;
  std::vector<uint8_t> active_;
  std::vector<faiss::idx_t> free_slots_;
  std::vector<faiss::idx_t> membership_;
  std::atomic<uint64_t> split_serial_{0};
  uint64_t split_count_{0};
  std::atomic<uint64_t> reassign_scanned_{0};
  uint64_t reassign_moved_{0};
  std::vector<std::unique_ptr<std::shared_mutex>> list_mutexes_;
  std::vector<uint64_t> list_generation_;
  std::atomic<faiss::idx_t> published_rows_{0};
  mutable std::mutex insert_mutex_;
  mutable std::shared_mutex state_mutex_;
  mutable std::mutex membership_mutex_;
  mutable std::mutex maintenance_queue_mutex_;
  std::condition_variable maintenance_queue_cv_;
  std::condition_variable maintenance_idle_cv_;
  std::deque<faiss::idx_t> maintenance_queue_;
  std::vector<uint8_t> maintenance_queued_or_running_;
  std::vector<double> maintenance_request_ms_;
  std::vector<std::thread> maintenance_workers_;
  bool maintenance_started_{false};
  bool maintenance_stopping_{false};
  size_t maintenance_active_workers_{0};
  std::exception_ptr maintenance_error_;
  mutable std::mutex maintenance_events_mutex_;
  std::vector<LireMaintenanceEvent> maintenance_events_;
  std::chrono::steady_clock::time_point maintenance_epoch_;
  std::atomic<uint64_t> maintenance_job_serial_{0};
};

}  // namespace baseline_eval
