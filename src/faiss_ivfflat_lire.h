#pragma once

#include <faiss/IndexIVFFlat.h>

#include <cstdint>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace baseline_eval {

// A reusable, fixed-size team for parallel work inside one logical query.
// The calling thread is team member 0; the remaining members are persistent
// helper threads created with the team.
class QuerySearchTeam final {
 public:
  explicit QuerySearchTeam(uint32_t thread_count);
  ~QuerySearchTeam();
  QuerySearchTeam(const QuerySearchTeam&) = delete;
  QuerySearchTeam& operator=(const QuerySearchTeam&) = delete;

  uint32_t thread_count() const;
  void parallel_for(size_t task_count,
                    const std::function<void(uint32_t, size_t)>& task);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

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
  std::string type{"lire_split"};
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
  uint64_t split_jobs_enqueued{0};
  uint64_t split_jobs_completed{0};
  uint64_t reassign_jobs_enqueued{0};
  uint64_t reassign_jobs_started{0};
  uint64_t reassign_jobs_completed{0};
  uint64_t reassign_jobs_dropped_stale{0};
  uint64_t reassign_jobs_noop_same_target{0};
  uint64_t reassign_vectors_moved{0};
  uint64_t stale_physical_copies_filtered{0};
};

struct LireOversizedListStats {
  uint64_t oversized_active_lists{0};
  uint64_t newly_oversized_lists{0};
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
  LireOversizedListStats sample_oversized_active_lists(
      bool reset_baseline = false) const;
  void search_at_frontier(faiss::idx_t n,
                          const float* x,
                          faiss::idx_t k,
                          float* distances,
                          faiss::idx_t* labels,
                          faiss::idx_t visible_rows,
                          QuerySearchTeam* query_team = nullptr) const;

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

  struct MaintenanceJob {
    enum class Kind { Split, Reassign };
    Kind kind{Kind::Split};
    faiss::idx_t triggering_list{-1};
    faiss::idx_t vector_id{-1};
    faiss::idx_t previous_slot{-1};
    double request_ms{0.0};
  };

  struct MaintenanceResult {
    uint32_t split_count{0};
    uint64_t stale_retries{0};
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
                                      uint64_t* stale_copies_filtered);
  void collect_and_reassign(
      const std::vector<float>& old_center,
      faiss::idx_t child0,
      faiss::idx_t child1,
      uint64_t* stale_copies_filtered);
  bool run_reassign(const MaintenanceJob& job,
                    bool* dropped_stale,
                    bool* noop_same_target,
                    faiss::idx_t* target_slot);
  MaintenanceResult run_maintenance(const std::vector<faiss::idx_t>& touched);
  MaintenanceResult run_maintenance_one(faiss::idx_t slot);
  void enqueue_maintenance(faiss::idx_t slot);
  void enqueue_reassign(MaintenanceJob job);
  void enqueue_job(MaintenanceJob job);
  void maintenance_worker();
  void note_oversized_list_state(faiss::idx_t slot,
                                 bool active,
                                 size_t list_size);
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
  uint64_t topology_generation_{0};
  std::atomic<uint64_t> split_serial_{0};
  uint64_t split_count_{0};
  std::atomic<uint64_t> reassign_scanned_{0};
  std::atomic<uint64_t> reassign_moved_{0};
  mutable std::atomic<uint64_t> stale_physical_copies_filtered_{0};
  std::vector<std::unique_ptr<std::shared_mutex>> list_mutexes_;
  std::atomic<faiss::idx_t> published_rows_{0};
  mutable std::mutex insert_mutex_;
  mutable std::shared_mutex state_mutex_;
  mutable std::shared_mutex membership_mutex_;
  mutable std::mutex oversized_stats_mutex_;
  mutable std::vector<uint8_t> previous_oversized_lists_;
  mutable std::vector<uint8_t> newly_oversized_since_sample_;
  mutable bool oversized_stats_initialized_{false};
  mutable std::mutex maintenance_queue_mutex_;
  std::condition_variable maintenance_queue_cv_;
  std::condition_variable maintenance_idle_cv_;
  std::deque<MaintenanceJob> maintenance_queue_;
  std::vector<uint8_t> maintenance_queued_or_running_;
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
  uint64_t split_jobs_enqueued_{0};
  uint64_t split_jobs_completed_{0};
  uint64_t reassign_jobs_enqueued_{0};
  uint64_t reassign_jobs_started_{0};
  uint64_t reassign_jobs_completed_{0};
  uint64_t reassign_jobs_dropped_stale_{0};
  uint64_t reassign_jobs_noop_same_target_{0};
  uint64_t reassign_vectors_moved_{0};
  std::exception_ptr maintenance_error_;
  mutable std::mutex maintenance_events_mutex_;
  std::vector<LireMaintenanceEvent> maintenance_events_;
  std::chrono::steady_clock::time_point maintenance_epoch_;
  std::chrono::steady_clock::time_point maintenance_last_log_;
  std::atomic<uint64_t> maintenance_job_serial_{0};
  mutable std::mutex maintenance_log_mutex_;
};

}  // namespace baseline_eval
