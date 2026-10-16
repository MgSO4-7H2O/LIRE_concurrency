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
#include <utility>
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
  double queue_wait_ms{0.0};
  double execution_ms{0.0};
  uint32_t affected_lists{0};
  std::vector<faiss::idx_t> affected_list_ids;
  uint32_t split_count{0};
  uint64_t reassigned_vectors{0};
  uint64_t stale_retries{0};
};

struct LireMaintenanceStats {
  uint64_t jobs_enqueued{0};
  uint64_t jobs_started{0};
  uint64_t jobs_completed{0};
  uint64_t jobs_retried{0};
  uint64_t pending_jobs{0};
  uint64_t running_jobs{0};
  uint64_t peak_pending_jobs{0};
  double total_queue_wait_ms{0.0};
  double total_execution_ms{0.0};
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
  bool wait_for_maintenance_backlog(uint64_t max_pending_jobs);
  void stop_maintenance_workers();
  void reset_maintenance_timeline();
  void validate_consistency();
  faiss::idx_t published_rows() const;
  std::vector<LireMaintenanceEvent> maintenance_events() const;
  LireMaintenanceStats maintenance_stats() const;
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
    uint64_t stale_retries{0};
    bool stale{false};
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
                                      uint64_t* reassigned_vectors,
                                      bool* stale,
                                      uint64_t* stale_retries);
  std::vector<faiss::idx_t> collect_and_reassign(
      faiss::idx_t old_slot,
      const std::vector<float>& old_center,
      faiss::idx_t child0,
      faiss::idx_t child1,
      uint64_t* reassigned_vectors,
      uint64_t* stale_retries);
  uint64_t apply_moves(
      const std::vector<Move>& moves,
      uint64_t topology_generation,
      const std::vector<std::pair<faiss::idx_t, uint64_t>>& expected_generations,
      bool* stale);
  MaintenanceResult run_maintenance(const std::vector<faiss::idx_t>& touched);
  MaintenanceResult run_maintenance_one(faiss::idx_t slot);
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
  uint64_t topology_generation_{0};
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
  uint64_t maintenance_jobs_enqueued_{0};
  uint64_t maintenance_jobs_started_{0};
  uint64_t maintenance_jobs_completed_{0};
  uint64_t maintenance_jobs_retried_{0};
  uint64_t maintenance_peak_pending_jobs_{0};
  double maintenance_total_queue_wait_ms_{0.0};
  double maintenance_total_execution_ms_{0.0};
  std::exception_ptr maintenance_error_;
  mutable std::mutex maintenance_events_mutex_;
  std::vector<LireMaintenanceEvent> maintenance_events_;
  std::chrono::steady_clock::time_point maintenance_epoch_;
  std::chrono::steady_clock::time_point maintenance_last_log_;
  std::atomic<uint64_t> maintenance_job_serial_{0};
  mutable std::mutex maintenance_log_mutex_;
};

}  // namespace baseline_eval
