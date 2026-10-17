#include "faiss_ivfflat_lire.h"

#include <faiss/IndexFlat.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace baseline_eval {

class QuerySearchTeam::Impl {
 public:
  explicit Impl(uint32_t count) : thread_count_(count) {
    if (thread_count_ == 0)
      throw std::runtime_error("query search team size must be positive");
    helpers_.reserve(thread_count_ - 1);
    try {
      for (uint32_t worker = 1; worker < thread_count_; ++worker)
        helpers_.emplace_back([this, worker] { helper_loop(worker); });
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        ++generation_;
      }
      start_cv_.notify_all();
      for (std::thread& helper : helpers_)
        if (helper.joinable()) helper.join();
      throw;
    }
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      ++generation_;
    }
    start_cv_.notify_all();
    for (std::thread& helper : helpers_)
      if (helper.joinable()) helper.join();
  }

  uint32_t thread_count() const { return thread_count_; }

  void parallel_for(
      size_t task_count,
      const std::function<void(uint32_t, size_t)>& task) {
    if (task_count == 0) return;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (busy_) throw std::runtime_error("query search team used concurrently");
      busy_ = true;
      task_ = task;
      task_count_ = task_count;
      pending_helpers_ = helpers_.size();
      task_error_ = nullptr;
      ++generation_;
    }
    start_cv_.notify_all();
    run_tasks(0);
    std::exception_ptr error;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [this] { return pending_helpers_ == 0; });
      error = task_error_;
      task_ = {};
      busy_ = false;
    }
    if (error) std::rethrow_exception(error);
  }

 private:
  void remember_error(std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!task_error_) task_error_ = error;
  }

  void run_tasks(uint32_t worker) {
    for (size_t task_index = worker; task_index < task_count_;
         task_index += thread_count_) {
      try {
        task_(worker, task_index);
      } catch (...) {
        remember_error(std::current_exception());
        return;
      }
    }
  }

  void helper_loop(uint32_t worker) {
    uint64_t observed_generation = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_cv_.wait(lock, [this, observed_generation] {
          return stopping_ || generation_ != observed_generation;
        });
        if (stopping_) return;
        observed_generation = generation_;
      }
      run_tasks(worker);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--pending_helpers_ == 0) done_cv_.notify_one();
      }
    }
  }

  const uint32_t thread_count_;
  std::vector<std::thread> helpers_;
  std::mutex mutex_;
  std::condition_variable start_cv_;
  std::condition_variable done_cv_;
  bool stopping_{false};
  bool busy_{false};
  uint64_t generation_{0};
  size_t pending_helpers_{0};
  size_t task_count_{0};
  std::function<void(uint32_t, size_t)> task_;
  std::exception_ptr task_error_;
};

QuerySearchTeam::QuerySearchTeam(uint32_t thread_count)
    : impl_(std::make_unique<Impl>(thread_count)) {}

QuerySearchTeam::~QuerySearchTeam() = default;

uint32_t QuerySearchTeam::thread_count() const {
  return impl_->thread_count();
}

void QuerySearchTeam::parallel_for(
    size_t task_count,
    const std::function<void(uint32_t, size_t)>& task) {
  impl_->parallel_for(task_count, task);
}

class IndexIVFFlatLIRE::ActiveCentroidIndex final : public faiss::Index {
 public:
  ActiveCentroidIndex(int d, uint32_t physical_nlist)
      : faiss::Index(d, faiss::METRIC_L2),
        dense_(d),
        physical_nlist_(physical_nlist),
        physical_centroids_(static_cast<size_t>(physical_nlist) * d, 0.0f) {
    is_trained = false;
  }

  void rebuild(const std::vector<float>& centroids,
               const std::vector<uint8_t>& active) {
    if (centroids.size() != physical_centroids_.size() ||
        active.size() != physical_nlist_) {
      throw std::runtime_error("invalid LIRE centroid state");
    }
    physical_centroids_ = centroids;
    dense_to_slot_.clear();
    std::vector<float> dense_centroids;
    dense_centroids.reserve(centroids.size());
    for (uint32_t slot = 0; slot < physical_nlist_; ++slot) {
      if (!active[slot]) continue;
      dense_to_slot_.push_back(static_cast<faiss::idx_t>(slot));
      const float* c = centroids.data() + static_cast<size_t>(slot) * d;
      dense_centroids.insert(dense_centroids.end(), c, c + d);
    }
    dense_.reset();
    if (!dense_to_slot_.empty()) {
      dense_.add(static_cast<faiss::idx_t>(dense_to_slot_.size()),
                 dense_centroids.data());
    }
    // IndexIVF checks quantizer->ntotal == nlist when it is constructed.
    // Search is implemented against dense_to_slot_, while ntotal reports the
    // fixed physical list capacity expected by the parent IndexIVF.
    ntotal = static_cast<faiss::idx_t>(physical_nlist_);
    is_trained = !dense_to_slot_.empty();
  }

  void add(faiss::idx_t, const float*) override {
    throw std::runtime_error("LIRE centroid index is rebuilt, not appended");
  }

  void search(faiss::idx_t n,
              const float* x,
              faiss::idx_t k,
              float* distances,
              faiss::idx_t* labels,
              const faiss::SearchParameters* params = nullptr) const override {
    (void)params;
    if (k <= 0) return;
    const faiss::idx_t active_count =
        static_cast<faiss::idx_t>(dense_to_slot_.size());
    const faiss::idx_t actual_k = std::min(k, active_count);
    std::fill(distances, distances + static_cast<size_t>(n) * k,
              std::numeric_limits<float>::infinity());
    std::fill(labels, labels + static_cast<size_t>(n) * k, -1);
    if (actual_k == 0) return;

    std::vector<float> dense_distances(static_cast<size_t>(n) * actual_k);
    std::vector<faiss::idx_t> dense_labels(static_cast<size_t>(n) * actual_k);
    dense_.search(n, x, actual_k, dense_distances.data(), dense_labels.data());
    for (faiss::idx_t i = 0; i < n; ++i) {
      for (faiss::idx_t j = 0; j < actual_k; ++j) {
        const size_t src = static_cast<size_t>(i) * actual_k + j;
        const size_t dst = static_cast<size_t>(i) * k + j;
        distances[dst] = dense_distances[src];
        labels[dst] = dense_to_slot_[static_cast<size_t>(dense_labels[src])];
      }
    }
  }

  void reset() override {
    dense_.reset();
    dense_to_slot_.clear();
    ntotal = 0;
    is_trained = false;
  }

  void reconstruct(faiss::idx_t key, float* recons) const override {
    if (key < 0 || key >= static_cast<faiss::idx_t>(physical_nlist_)) {
      throw std::runtime_error("invalid LIRE centroid slot");
    }
    const float* src = physical_centroids_.data() + static_cast<size_t>(key) * d;
    std::copy(src, src + d, recons);
  }

 private:
  faiss::IndexFlatL2 dense_;
  uint32_t physical_nlist_;
  std::vector<faiss::idx_t> dense_to_slot_;
  std::vector<float> physical_centroids_;
};

IndexIVFFlatLIRE::IndexIVFFlatLIRE(int d, const LireConfig& config)
    : faiss::IndexIVFFlat(
          new ActiveCentroidIndex(d, config.max_nlist),
          static_cast<size_t>(d),
          static_cast<size_t>(config.max_nlist),
          faiss::METRIC_L2),
      config_(config),
      initial_nlist_(config.initial_nlist),
      max_nlist_(config.max_nlist),
      centroids_(static_cast<size_t>(config.max_nlist) * d, 0.0f),
      active_(config.max_nlist, 0),
      maintenance_queued_or_running_(config.max_nlist, 0) {
  if (d <= 0 || initial_nlist_ == 0 || max_nlist_ < initial_nlist_ ||
      config_.split_threshold < 2) {
    throw std::runtime_error("invalid IndexIVFFlatLIRE configuration");
  }
  own_fields = true;
  is_trained = false;
  list_mutexes_.reserve(max_nlist_);
  for (uint32_t i = 0; i < max_nlist_; ++i) {
    list_mutexes_.push_back(std::make_unique<std::shared_mutex>());
  }
  maintenance_epoch_ = std::chrono::steady_clock::now();
  maintenance_last_log_ = maintenance_epoch_;
}

IndexIVFFlatLIRE::~IndexIVFFlatLIRE() {
  try {
    stop_maintenance_workers();
  } catch (...) {
  }
}

std::shared_mutex& IndexIVFFlatLIRE::list_mutex(faiss::idx_t slot) const {
  if (slot < 0 || slot >= static_cast<faiss::idx_t>(max_nlist_)) {
    throw std::runtime_error("invalid LIRE list slot");
  }
  return *list_mutexes_[static_cast<size_t>(slot)];
}

IndexIVFFlatLIRE::ActiveCentroidIndex*
IndexIVFFlatLIRE::active_quantizer() {
  return static_cast<ActiveCentroidIndex*>(quantizer);
}

const IndexIVFFlatLIRE::ActiveCentroidIndex*
IndexIVFFlatLIRE::active_quantizer() const {
  return static_cast<const ActiveCentroidIndex*>(quantizer);
}

float IndexIVFFlatLIRE::l2(const float* a, const float* b) const {
  return faiss::fvec_L2sqr(a, b, static_cast<size_t>(d));
}

const float* IndexIVFFlatLIRE::centroid(faiss::idx_t slot) const {
  return centroids_.data() + static_cast<size_t>(slot) * d;
}

float* IndexIVFFlatLIRE::centroid(faiss::idx_t slot) {
  return centroids_.data() + static_cast<size_t>(slot) * d;
}

IndexIVFFlatLIRE::ListData IndexIVFFlatLIRE::read_list(
    faiss::idx_t slot) const {
  ListData out;
  const size_t size = invlists->list_size(static_cast<size_t>(slot));
  out.ids.resize(size);
  out.vectors.resize(size * static_cast<size_t>(d));
  if (size == 0) return out;

  const faiss::idx_t* ids = invlists->get_ids(static_cast<size_t>(slot));
  const uint8_t* codes = invlists->get_codes(static_cast<size_t>(slot));
  std::copy(ids, ids + size, out.ids.begin());
  std::memcpy(out.vectors.data(), codes,
              size * static_cast<size_t>(d) * sizeof(float));
  invlists->release_ids(static_cast<size_t>(slot), ids);
  invlists->release_codes(static_cast<size_t>(slot), codes);
  return out;
}

void IndexIVFFlatLIRE::write_list(faiss::idx_t slot, const ListData& list) {
  if (list.vectors.size() != list.ids.size() * static_cast<size_t>(d)) {
    throw std::runtime_error("invalid LIRE list payload");
  }
  invlists->resize(static_cast<size_t>(slot), 0);
  if (!list.ids.empty()) {
    invlists->add_entries(
        static_cast<size_t>(slot),
        list.ids.size(),
        list.ids.data(),
        reinterpret_cast<const uint8_t*>(list.vectors.data()));
  }
}

void IndexIVFFlatLIRE::rebuild_quantizer() {
  active_quantizer()->rebuild(centroids_, active_);
}

faiss::idx_t IndexIVFFlatLIRE::nearest_active(const float* vector) const {
  faiss::idx_t best = -1;
  float best_distance = std::numeric_limits<float>::infinity();
  for (uint32_t slot = 0; slot < max_nlist_; ++slot) {
    if (!active_[slot]) continue;
    const float distance = l2(vector, centroid(slot));
    if (distance < best_distance ||
        (distance == best_distance && static_cast<faiss::idx_t>(slot) < best)) {
      best_distance = distance;
      best = static_cast<faiss::idx_t>(slot);
    }
  }
  return best;
}

std::vector<faiss::idx_t> IndexIVFFlatLIRE::nearest_centroid_slots(
    const float* center,
    faiss::idx_t excluded,
    uint32_t k) const {
  std::vector<std::pair<float, faiss::idx_t>> candidates;
  candidates.reserve(active_nlist_);
  for (uint32_t slot = 0; slot < max_nlist_; ++slot) {
    if (!active_[slot] || static_cast<faiss::idx_t>(slot) == excluded) continue;
    candidates.emplace_back(l2(center, centroid(slot)),
                            static_cast<faiss::idx_t>(slot));
  }
  const size_t keep = std::min<size_t>(k, candidates.size());
  std::partial_sort(candidates.begin(), candidates.begin() + keep,
                    candidates.end(),
                    [](const auto& a, const auto& b) {
                      return a.first < b.first ||
                             (a.first == b.first && a.second < b.second);
                    });
  std::vector<faiss::idx_t> out;
  out.reserve(keep);
  for (size_t i = 0; i < keep; ++i) out.push_back(candidates[i].second);
  return out;
}

bool IndexIVFFlatLIRE::split_balanced_kmeans(
    const ListData& input,
    std::vector<uint8_t>* labels,
    std::vector<float>* centers) {
  const size_t n = input.size();
  if (n < 2) return false;

  // This is the CPU/L2 specialization of SPFresh's Split() call to
  // KmeansClustering(K=2, samples=1000, lambdaFactor=100). The assignment
  // uses the same count penalty: L2(x, c_k) + lambda * previous_count[k].
  const uint64_t serial = split_serial_.fetch_add(1, std::memory_order_relaxed);
  std::mt19937 rng(config_.seed + static_cast<uint32_t>(serial));
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  const size_t sample_n = std::min<size_t>(1000, n);

  std::vector<float> best_centers(static_cast<size_t>(2) * d, 0.0f);
  float best_objective = std::numeric_limits<float>::infinity();
  float adjusted_lambda = 0.0f;
  size_t best_counts[2] = {0, 0};

  for (int trial = 0; trial < 3; ++trial) {
    std::shuffle(order.begin(), order.end(), rng);
    size_t a = order[0];
    size_t b = order[1];
    if (a == b) b = order.back();
    std::vector<float> trial_centers(static_cast<size_t>(2) * d);
    std::copy(input.row(a, d), input.row(a, d) + d, trial_centers.begin());
    std::copy(input.row(b, d), input.row(b, d) + d,
              trial_centers.begin() + d);

    size_t counts[2] = {0, 0};
    float sums[2] = {0.0f, 0.0f};
    float max_dist[2] = {0.0f, 0.0f};
    float objective = 0.0f;
    for (size_t oi = 0; oi < sample_n; ++oi) {
      const float* vector = input.row(order[oi], d);
      const float d0 = l2(vector, trial_centers.data());
      const float d1 = l2(vector, trial_centers.data() + d);
      const int label = d1 < d0 ? 1 : 0;
      const float distance = label == 0 ? d0 : d1;
      counts[label]++;
      sums[label] += distance;
      max_dist[label] = std::max(max_dist[label], distance);
      objective += distance;
    }
    if (counts[0] == 0 || counts[1] == 0) continue;
    if (objective < best_objective) {
      best_objective = objective;
      best_centers = trial_centers;
      const int larger = counts[1] > counts[0] ? 1 : 0;
      adjusted_lambda =
          (max_dist[larger] - sums[larger] / counts[larger]) /
          static_cast<float>(sample_n);
      if (adjusted_lambda < 0.0f) adjusted_lambda = 0.0f;
      best_counts[0] = counts[0];
      best_counts[1] = counts[1];
    }
  }
  if (!std::isfinite(best_objective)) return false;

  std::vector<float> current = best_centers;
  std::vector<float> next(static_cast<size_t>(2) * d, 0.0f);
  size_t previous_counts[2] = {best_counts[0], best_counts[1]};
  float best_iter_objective = std::numeric_limits<float>::infinity();
  int no_improvement = 0;
  const float original_lambda = 1.0f /
      (100.0f * static_cast<float>(std::max<size_t>(1, sample_n)));
  const float lambda = std::min(adjusted_lambda, original_lambda);

  for (int iteration = 0; iteration < 100; ++iteration) {
    std::shuffle(order.begin(), order.end(), rng);
    std::fill(next.begin(), next.end(), 0.0f);
    size_t counts[2] = {0, 0};
    float farthest_distance[2] = {-1.0f, -1.0f};
    size_t farthest_index[2] = {0, 0};
    float objective = 0.0f;
    for (size_t oi = 0; oi < sample_n; ++oi) {
      const size_t index = order[oi];
      const float* vector = input.row(index, d);
      const float d0 = l2(vector, current.data()) +
          lambda * static_cast<float>(previous_counts[0]);
      const float d1 = l2(vector, current.data() + d) +
          lambda * static_cast<float>(previous_counts[1]);
      const int label = d1 < d0 ? 1 : 0;
      const float distance = label == 0 ? d0 : d1;
      counts[label]++;
      objective += distance;
      if (distance > farthest_distance[label]) {
        farthest_distance[label] = distance;
        farthest_index[label] = index;
      }
      float* sum = next.data() + static_cast<size_t>(label) * d;
      for (int j = 0; j < d; ++j) sum[j] += vector[j];
    }
    int max_cluster = -1;
    size_t max_previous_count = 0;
    for (int label = 0; label < 2; ++label) {
      if (counts[label] > 0 && previous_counts[label] > max_previous_count) {
        max_cluster = label;
        max_previous_count = previous_counts[label];
      }
    }
    for (int label = 0; label < 2; ++label) {
      float* center_out = next.data() + static_cast<size_t>(label) * d;
      if (counts[label] == 0) {
        if (max_cluster < 0) return false;
        const float* replacement = input.row(farthest_index[max_cluster], d);
        std::copy(replacement, replacement + d, center_out);
      } else {
        for (int j = 0; j < d; ++j) {
          center_out[j] /= static_cast<float>(counts[label]);
        }
      }
    }

    float center_diff = l2(current.data(), next.data()) +
                        l2(current.data() + d, next.data() + d);
    current.swap(next);
    previous_counts[0] = counts[0];
    previous_counts[1] = counts[1];
    if (objective < best_iter_objective) {
      best_iter_objective = objective;
      no_improvement = 0;
    } else {
      no_improvement++;
    }
    if (center_diff < 1e-3f || no_improvement >= 5) break;
  }

  // SPFresh performs one full, unpenalized assignment, replaces each
  // virtual center with the nearest vector in that cluster, then assigns once
  // more using those posting heads.
  size_t nearest_index[2] = {0, 0};
  float nearest_distance[2] = {
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity()};
  size_t first_counts[2] = {0, 0};
  for (size_t i = 0; i < n; ++i) {
    const float* vector = input.row(i, d);
    const float d0 = l2(vector, current.data());
    const float d1 = l2(vector, current.data() + d);
    const int label = d1 < d0 ? 1 : 0;
    const float distance = label == 0 ? d0 : d1;
    first_counts[label]++;
    if (distance <= nearest_distance[label]) {
      nearest_distance[label] = distance;
      nearest_index[label] = i;
    }
  }
  if (first_counts[0] == 0 || first_counts[1] == 0) return false;
  for (int label = 0; label < 2; ++label) {
    const float* head = input.row(nearest_index[label], d);
    std::copy(head, head + d, current.data() + static_cast<size_t>(label) * d);
  }

  labels->assign(n, 0);
  size_t counts[2] = {0, 0};
  for (size_t i = 0; i < n; ++i) {
    const float* vector = input.row(i, d);
    const uint8_t label =
        l2(vector, current.data() + d) < l2(vector, current.data()) ? 1 : 0;
    (*labels)[i] = label;
    counts[label]++;
  }
  if (counts[0] == 0 || counts[1] == 0) return false;
  *centers = std::move(current);
  return true;
}

std::vector<faiss::idx_t> IndexIVFFlatLIRE::split_one(
    faiss::idx_t slot,
    bool* did_split,
    uint64_t* stale_copies_filtered) {
  *did_split = false;
  *stale_copies_filtered = 0;

  // Reserve an inactive slot before taking posting locks. This lets the
  // eventual source/destination lock set be acquired in ascending order.
  faiss::idx_t new_slot = -1;
  {
    std::unique_lock<std::shared_mutex> state_lock(state_mutex_);
    if (!active_[static_cast<size_t>(slot)]) return {};
    if (free_slots_.empty())
      throw std::runtime_error("LIRE max_nlist exhausted during split");
    new_slot = free_slots_.back();
    free_slots_.pop_back();
  }

  auto release_reserved_slot = [&] {
    std::unique_lock<std::shared_mutex> state_lock(state_mutex_);
    if (!active_[static_cast<size_t>(new_slot)])
      free_slots_.push_back(new_slot);
  };
  const auto reservation_guard = std::shared_ptr<void>(
      nullptr, [&](void*) { release_reserved_slot(); });

  const std::vector<faiss::idx_t> lock_slots =
      slot < new_slot ? std::vector<faiss::idx_t>{slot, new_slot}
                      : std::vector<faiss::idx_t>{new_slot, slot};
  std::vector<std::unique_lock<std::shared_mutex>> list_locks;
  list_locks.reserve(lock_slots.size());
  for (faiss::idx_t list : lock_slots)
    list_locks.emplace_back(list_mutex(list));

  ListData input;
  std::vector<float> old_center;
  bool should_split = false;
  {
    std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
    should_split = active_[static_cast<size_t>(slot)] &&
        !active_[static_cast<size_t>(new_slot)] &&
        invlists->list_size(static_cast<size_t>(slot)) >
            config_.split_threshold;
    if (should_split) {
      old_center.assign(centroid(slot), centroid(slot) + d);
      const ListData physical = read_list(slot);
      std::shared_lock<std::shared_mutex> membership_lock(membership_mutex_);
      input.ids.reserve(physical.size());
      input.vectors.reserve(physical.vectors.size());
      for (size_t i = 0; i < physical.size(); ++i) {
        const faiss::idx_t id = physical.ids[i];
        if (id < 0 || static_cast<size_t>(id) >= membership_.size() ||
            membership_[static_cast<size_t>(id)] != slot) {
          ++*stale_copies_filtered;
          continue;
        }
        input.ids.push_back(id);
        const float* vector = physical.row(i, d);
        input.vectors.insert(input.vectors.end(), vector, vector + d);
      }
    }
  }

  if (!should_split) {
    return {};
  }

  if (input.size() <= config_.split_threshold) {
    // Reclaim old physical copies even when their removal leaves the posting
    // below the split trigger.
    write_list(slot, input);
    note_oversized_list_state(slot, true, input.size());
    maintenance_stale_copies_filtered_.fetch_add(
        *stale_copies_filtered, std::memory_order_relaxed);
    return {slot};
  }

  std::vector<uint8_t> labels;
  std::vector<float> new_centers;
  if (!split_balanced_kmeans(input, &labels, &new_centers)) {
    maintenance_stale_copies_filtered_.fetch_add(
        *stale_copies_filtered, std::memory_order_relaxed);
    return {};
  }

  ListData children[2];
  children[0].ids.reserve(input.size());
  children[1].ids.reserve(input.size());
  children[0].vectors.reserve(input.vectors.size());
  children[1].vectors.reserve(input.vectors.size());
  for (size_t i = 0; i < input.size(); ++i) {
    const int label = labels[i];
    children[label].ids.push_back(input.ids[i]);
    const float* vector = input.row(i, d);
    children[label].vectors.insert(children[label].vectors.end(),
                                   vector, vector + d);
  }

  {
    // Split owns the current source posting for the entire read/compute/commit
    // interval. The only global exclusive section is topology publication.
    // Lock order is posting IDs ascending, then state_mutex_, then membership.
    std::unique_lock<std::shared_mutex> state_lock(state_mutex_);
    if (!active_[static_cast<size_t>(slot)] ||
        active_[static_cast<size_t>(new_slot)]) {
      maintenance_stale_copies_filtered_.fetch_add(
          *stale_copies_filtered, std::memory_order_relaxed);
      return {};
    }
    std::unique_lock<std::shared_mutex> membership_lock(membership_mutex_);
    write_list(slot, children[0]);
    write_list(new_slot, children[1]);
    std::copy(new_centers.begin(), new_centers.begin() + d, centroid(slot));
    std::copy(new_centers.begin() + d, new_centers.end(), centroid(new_slot));
    active_[static_cast<size_t>(new_slot)] = 1;
    active_nlist_++;
    for (faiss::idx_t id : children[0].ids)
      membership_[static_cast<size_t>(id)] = slot;
    for (faiss::idx_t id : children[1].ids)
      membership_[static_cast<size_t>(id)] = new_slot;
    ++topology_generation_;
    note_oversized_list_state(slot, true, children[0].size());
    note_oversized_list_state(new_slot, true, children[1].size());
    ++split_count_;
    rebuild_quantizer();
  }
  *did_split = true;
  list_locks.clear();

  std::vector<faiss::idx_t> changed{slot, new_slot};
  if (config_.enable_reassign) {
    collect_and_reassign(
        old_center, slot, new_slot, stale_copies_filtered);
  }
  maintenance_stale_copies_filtered_.fetch_add(
      *stale_copies_filtered, std::memory_order_relaxed);
  return changed;
}

void IndexIVFFlatLIRE::collect_and_reassign(
    const std::vector<float>& old_center,
    faiss::idx_t child0,
    faiss::idx_t child1,
    uint64_t* stale_copies_filtered) {
  std::vector<float> centroid_snapshot;
  std::vector<uint8_t> active_snapshot;
  std::vector<faiss::idx_t> neighbors;
  {
    std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
    if (!active_[static_cast<size_t>(child0)] ||
        !active_[static_cast<size_t>(child1)]) return;
    centroid_snapshot = centroids_;
    active_snapshot = active_;
    neighbors = nearest_centroid_slots(old_center.data(), child0,
                                       config_.reassign_k + 1);
  }
  neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), child0),
                  neighbors.end());
  neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), child1),
                  neighbors.end());
  if (neighbors.size() > config_.reassign_k)
    neighbors.resize(config_.reassign_k);

  std::vector<faiss::idx_t> lists{child0, child1};
  lists.insert(lists.end(), neighbors.begin(), neighbors.end());
  std::sort(lists.begin(), lists.end());
  lists.erase(std::unique(lists.begin(), lists.end()), lists.end());
  std::vector<std::shared_lock<std::shared_mutex>> list_locks;
  list_locks.reserve(lists.size());
  for (faiss::idx_t list : lists) list_locks.emplace_back(list_mutex(list));
  std::shared_lock<std::shared_mutex> membership_lock(membership_mutex_);

  const auto center_for = [&](faiss::idx_t list) {
    return centroid_snapshot.data() + static_cast<size_t>(list) * d;
  };
  std::unordered_set<faiss::idx_t> candidate_ids;
  auto scan_list = [&](faiss::idx_t list, bool is_child) {
    const ListData data = read_list(list);
    for (size_t offset = 0; offset < data.size(); ++offset) {
      const faiss::idx_t id = data.ids[offset];
      if (id < 0 || static_cast<size_t>(id) >= membership_.size() ||
          membership_[static_cast<size_t>(id)] != list) {
        ++*stale_copies_filtered;
        continue;
      }
      const float* vector = data.row(offset, d);
      reassign_scanned_.fetch_add(1, std::memory_order_relaxed);
      bool candidate = false;
      if (is_child) {
        candidate = l2(vector, old_center.data()) < l2(vector, center_for(list));
      } else {
        const float old_distance = l2(vector, old_center.data());
        const float child0_distance = l2(vector, center_for(child0));
        const float child1_distance = l2(vector, center_for(child1));
        const float current_distance = l2(vector, center_for(list));
        candidate = !((old_distance <= child0_distance &&
                       old_distance <= child1_distance) ||
                      (current_distance <= child0_distance &&
                       current_distance <= child1_distance));
      }
      if (!candidate || !candidate_ids.insert(id).second) continue;
      MaintenanceJob job;
      job.kind = MaintenanceJob::Kind::Reassign;
      job.triggering_list = list;
      job.vector_id = id;
      job.previous_slot = list;
      // This evaluator is insert-only: vector contents never change in place,
      // so the captured payload remains valid while the asynchronous job waits.
      job.vector_payload.assign(vector, vector + d);
      enqueue_reassign(std::move(job));
    }
  };

  scan_list(child0, true);
  scan_list(child1, true);
  for (faiss::idx_t neighbor : neighbors) {
    if (!active_snapshot[static_cast<size_t>(neighbor)] ||
        neighbor == child0 || neighbor == child1) continue;
    scan_list(neighbor, false);
  }
}

bool IndexIVFFlatLIRE::run_reassign(
    const MaintenanceJob& job,
    bool* dropped_stale,
    bool* noop_same_target,
    faiss::idx_t* target_slot) {
  *dropped_stale = false;
  *noop_same_target = false;
  *target_slot = -1;
  if (job.vector_id < 0 || job.previous_slot < 0 ||
      job.previous_slot >= static_cast<faiss::idx_t>(max_nlist_)) {
    *dropped_stale = true;
    return false;
  }

  if (job.vector_payload.size() != static_cast<size_t>(d))
    throw std::runtime_error("LIRE reassignment payload dimension mismatch");

  // Membership is the authoritative logical location. A stale job is dropped
  // here without searching its old posting or repeating candidate discovery.
  {
    std::shared_lock<std::shared_mutex> membership_lock(membership_mutex_);
    if (static_cast<size_t>(job.vector_id) >= membership_.size() ||
        membership_[static_cast<size_t>(job.vector_id)] != job.previous_slot) {
      *dropped_stale = true;
      return false;
    }
  }
  const float* vector = job.vector_payload.data();

  for (;;) {
    faiss::idx_t target = -1;
    uint64_t route_generation = 0;
    {
      std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
      if (!active_[static_cast<size_t>(job.previous_slot)]) {
        *dropped_stale = true;
        return false;
      }
      target = nearest_active(vector);
      route_generation = topology_generation_;
    }
    if (target < 0) {
      *dropped_stale = true;
      return false;
    }
    *target_slot = target;
    if (target == job.previous_slot) {
      std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
      if (route_generation != topology_generation_ ||
          nearest_active(vector) != target) continue;
      std::shared_lock<std::shared_mutex> membership_lock(membership_mutex_);
      if (static_cast<size_t>(job.vector_id) >= membership_.size() ||
          membership_[static_cast<size_t>(job.vector_id)] != job.previous_slot) {
        *dropped_stale = true;
      } else {
        *noop_same_target = true;
      }
      return false;
    }

    const faiss::idx_t first = std::min(job.previous_slot, target);
    const faiss::idx_t second = std::max(job.previous_slot, target);
    std::unique_lock<std::shared_mutex> first_lock(list_mutex(first));
    std::unique_lock<std::shared_mutex> second_lock(list_mutex(second));
    // The posting locks are always acquired before the state lock throughout
    // update and maintenance. A topology change while waiting only reroutes
    // this vector; it never causes neighborhood collection to be repeated.
    std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
    if (route_generation != topology_generation_) continue;
    if (!active_[static_cast<size_t>(job.previous_slot)] ||
        !active_[static_cast<size_t>(target)] ||
        nearest_active(vector) != target) continue;

    std::unique_lock<std::shared_mutex> membership_lock(membership_mutex_);
    if (static_cast<size_t>(job.vector_id) >= membership_.size() ||
        membership_[static_cast<size_t>(job.vector_id)] != job.previous_slot) {
      *dropped_stale = true;
      return false;
    }

    // Revive an existing stale physical copy when the vector returns to a
    // posting; otherwise append before publishing its authoritative location.
    bool already_present = false;
    const size_t target_size = invlists->list_size(static_cast<size_t>(target));
    if (target_size != 0) {
      const faiss::idx_t* ids = invlists->get_ids(static_cast<size_t>(target));
      for (size_t i = 0; i < target_size; ++i) {
        if (ids[i] == job.vector_id) {
          already_present = true;
          break;
        }
      }
      invlists->release_ids(static_cast<size_t>(target), ids);
    }
    if (!already_present) {
      const faiss::idx_t id = job.vector_id;
      invlists->add_entries(
          static_cast<size_t>(target), 1, &id,
          reinterpret_cast<const uint8_t*>(vector));
    }
    membership_[static_cast<size_t>(job.vector_id)] = target;
    note_oversized_list_state(
        target, true, invlists->list_size(static_cast<size_t>(target)));
    reassign_moved_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
}

IndexIVFFlatLIRE::MaintenanceResult IndexIVFFlatLIRE::run_maintenance(
    const std::vector<faiss::idx_t>& touched) {
  for (faiss::idx_t slot : touched) enqueue_maintenance(slot);
  start_maintenance_workers();
  wait_for_maintenance();
  return {};
}

IndexIVFFlatLIRE::MaintenanceResult IndexIVFFlatLIRE::run_maintenance_one(
    faiss::idx_t slot) {
  MaintenanceResult result;
  bool oversized = false;
  {
    std::shared_lock<std::shared_mutex> list_lock(list_mutex(slot));
    std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
    if (!active_[static_cast<size_t>(slot)]) return result;
    oversized = invlists->list_size(static_cast<size_t>(slot)) >
        config_.split_threshold;
  }
  if (!oversized) return result;
  bool did_split = false;
  uint64_t filtered = 0;
  result.affected_lists = split_one(slot, &did_split, &filtered);
  result.split_count = did_split ? 1u : 0u;
  return result;
}

void IndexIVFFlatLIRE::enqueue_maintenance(faiss::idx_t slot) {
  if (slot < 0 || slot >= static_cast<faiss::idx_t>(max_nlist_)) return;
  bool oversized = false;
  {
    std::shared_lock<std::shared_mutex> list_lock(list_mutex(slot));
    std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
    oversized = active_[static_cast<size_t>(slot)] &&
        invlists->list_size(static_cast<size_t>(slot)) > config_.split_threshold;
  }
  if (!oversized) return;
  MaintenanceJob job;
  job.kind = MaintenanceJob::Kind::Split;
  job.triggering_list = slot;
  enqueue_job(std::move(job));
}

void IndexIVFFlatLIRE::enqueue_reassign(MaintenanceJob job) {
  job.kind = MaintenanceJob::Kind::Reassign;
  enqueue_job(std::move(job));
}

void IndexIVFFlatLIRE::enqueue_job(MaintenanceJob job) {
  job.request_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - maintenance_epoch_).count();
  std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
  if (maintenance_stopping_) return;
  if (job.kind == MaintenanceJob::Kind::Split) {
    const faiss::idx_t slot = job.triggering_list;
    if (slot < 0 || slot >= static_cast<faiss::idx_t>(max_nlist_) ||
        maintenance_queued_or_running_[static_cast<size_t>(slot)]) return;
    maintenance_queued_or_running_[static_cast<size_t>(slot)] = 1;
    ++split_jobs_enqueued_;
  } else {
    ++reassign_jobs_enqueued_;
    reassign_payload_bytes_enqueued_ +=
        job.vector_payload.size() * sizeof(float);
  }
  ++maintenance_jobs_enqueued_;
  maintenance_queue_.push_back(std::move(job));
  maintenance_peak_pending_jobs_ = std::max<uint64_t>(
      maintenance_peak_pending_jobs_, maintenance_queue_.size());
  maintenance_queue_cv_.notify_one();
}

void IndexIVFFlatLIRE::maintenance_worker() {
  for (;;) {
    MaintenanceJob job;
    {
      std::unique_lock<std::mutex> lock(maintenance_queue_mutex_);
      maintenance_queue_cv_.wait(lock, [&] {
        return maintenance_stopping_ || !maintenance_queue_.empty();
      });
      if (maintenance_stopping_ && maintenance_queue_.empty()) return;
      job = std::move(maintenance_queue_.front());
      maintenance_queue_.pop_front();
      ++maintenance_active_workers_;
      ++maintenance_jobs_started_;
      if (job.kind == MaintenanceJob::Kind::Reassign)
        ++reassign_jobs_started_;
      maintenance_idle_cv_.notify_all();
    }

    LireMaintenanceEvent event;
    event.type = job.kind == MaintenanceJob::Kind::Split
        ? "lire_split" : "lire_reassign";
    event.triggering_list = job.kind == MaintenanceJob::Kind::Split
        ? job.triggering_list : job.previous_slot;
    event.request_ms = job.request_ms;
    bool completed = false;
    MaintenanceResult result;
    bool dropped_stale = false;
    bool noop_same_target = false;
    bool reassigned = false;
    faiss::idx_t target_slot = -1;
    try {
      event.job_id = maintenance_job_serial_.fetch_add(1, std::memory_order_relaxed) + 1;
      event.start_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - maintenance_epoch_).count();
      event.queue_wait_ms = std::max(0.0, event.start_ms - event.request_ms);
      if (job.kind == MaintenanceJob::Kind::Split) {
        result = run_maintenance_one(job.triggering_list);
      } else {
        reassigned = run_reassign(
            job, &dropped_stale, &noop_same_target, &target_slot);
      }
      event.compute_done_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - maintenance_epoch_).count();
      event.commit_done_ms = event.compute_done_ms;
      event.end_ms = event.compute_done_ms;
      event.execution_ms = std::max(0.0, event.end_ms - event.start_ms);
      event.split_count = result.split_count;
      event.reassigned_vectors = reassigned ? 1 : 0;
      event.stale_retries = result.stale_retries;
      event.affected_lists = static_cast<uint32_t>(result.affected_lists.size());
      event.affected_list_ids = result.affected_lists;
      if (job.kind == MaintenanceJob::Kind::Split) {
        std::lock_guard<std::mutex> event_lock(maintenance_events_mutex_);
        maintenance_events_.push_back(event);
      }
      completed = true;
    } catch (...) {
      std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
      if (!maintenance_error_) maintenance_error_ = std::current_exception();
    }

    {
      std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
      if (job.kind == MaintenanceJob::Kind::Split)
        maintenance_queued_or_running_[static_cast<size_t>(job.triggering_list)] = 0;
      if (completed) {
        ++maintenance_jobs_completed_;
        maintenance_jobs_retried_ += result.stale_retries;
        maintenance_total_queue_wait_ms_ += event.queue_wait_ms;
        maintenance_total_execution_ms_ += event.execution_ms;
        if (job.kind == MaintenanceJob::Kind::Split)
          split_jobs_completed_++;
        else {
          reassign_jobs_completed_++;
          reassign_jobs_dropped_stale_ += dropped_stale ? 1 : 0;
          reassign_jobs_noop_same_target_ += noop_same_target ? 1 : 0;
          reassign_vectors_moved_ += reassigned ? 1 : 0;
        }
      }
    }
    if (completed) {
      if (job.kind == MaintenanceJob::Kind::Split) {
        std::vector<faiss::idx_t> candidates = result.affected_lists;
        candidates.push_back(job.triggering_list);
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()),
                         candidates.end());
        for (faiss::idx_t candidate : candidates) enqueue_maintenance(candidate);
      } else if (reassigned && target_slot >= 0) {
        enqueue_maintenance(target_slot);
      }
    }
    {
      std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
      --maintenance_active_workers_;
      const auto now = std::chrono::steady_clock::now();
      if (completed && (maintenance_jobs_completed_ % 1000 == 0 ||
          now - maintenance_last_log_ >= std::chrono::seconds(1))) {
        maintenance_last_log_ = now;
        std::lock_guard<std::mutex> log_lock(maintenance_log_mutex_);
        std::cout << "[MAINTENANCE_PROGRESS] enqueued=" << maintenance_jobs_enqueued_
                  << " started=" << maintenance_jobs_started_
                  << " running=" << maintenance_active_workers_
                  << " completed=" << maintenance_jobs_completed_
                  << " pending=" << maintenance_queue_.size()
                  << " peak_pending=" << maintenance_peak_pending_jobs_
                  << " split_enqueued=" << split_jobs_enqueued_
                  << " split_completed=" << split_jobs_completed_
                  << " reassign_enqueued=" << reassign_jobs_enqueued_
                  << " reassign_started=" << reassign_jobs_started_
                  << " reassign_completed=" << reassign_jobs_completed_
                  << " reassign_stale_drops=" << reassign_jobs_dropped_stale_
                  << " reassign_moved=" << reassign_vectors_moved_
                  << " stale_retries=" << maintenance_jobs_retried_ << std::endl;
      }
      if (maintenance_queue_.empty() && maintenance_active_workers_ == 0)
        maintenance_idle_cv_.notify_all();
      else maintenance_idle_cv_.notify_all();
    }
  }
}

void IndexIVFFlatLIRE::start_maintenance_workers() {
  std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
  if (maintenance_started_) return;
  maintenance_stopping_ = false;
  maintenance_started_ = true;
  const uint32_t count = std::max<uint32_t>(1, config_.maintenance_threads);
  maintenance_workers_.reserve(count);
  for (uint32_t i = 0; i < count; ++i)
    maintenance_workers_.emplace_back(&IndexIVFFlatLIRE::maintenance_worker, this);
  maintenance_queue_cv_.notify_all();
}

void IndexIVFFlatLIRE::wait_for_maintenance() {
  bool needs_workers = false;
  {
    std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
    needs_workers = !maintenance_queue_.empty() && !maintenance_started_;
  }
  if (needs_workers) start_maintenance_workers();
  std::unique_lock<std::mutex> lock(maintenance_queue_mutex_);
  maintenance_idle_cv_.wait(lock, [&] {
    return maintenance_queue_.empty() && maintenance_active_workers_ == 0;
  });
  const std::exception_ptr error = maintenance_error_;
  lock.unlock();
  if (error) std::rethrow_exception(error);
}

bool IndexIVFFlatLIRE::wait_for_maintenance_backlog(uint64_t max_pending_jobs) {
  bool needs_workers = false;
  {
    std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
    needs_workers = !maintenance_queue_.empty() && !maintenance_started_;
  }
  if (needs_workers) start_maintenance_workers();
  std::unique_lock<std::mutex> lock(maintenance_queue_mutex_);
  const bool backpressured = maintenance_queue_.size() > max_pending_jobs;
  maintenance_idle_cv_.wait(lock, [&] {
    return maintenance_queue_.size() <= max_pending_jobs || maintenance_error_;
  });
  const std::exception_ptr error = maintenance_error_;
  lock.unlock();
  if (error) std::rethrow_exception(error);
  return backpressured;
}

void IndexIVFFlatLIRE::stop_maintenance_workers() {
  std::exception_ptr pending_error;
  try {
    wait_for_maintenance();
  } catch (...) {
    pending_error = std::current_exception();
  }
  {
    std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
    if (!maintenance_started_) {
      if (pending_error) std::rethrow_exception(pending_error);
      return;
    }
  }
  {
    std::unique_lock<std::mutex> lock(maintenance_queue_mutex_);
    maintenance_idle_cv_.wait(lock, [&] {
      return maintenance_queue_.empty() && maintenance_active_workers_ == 0;
    });
    if (!pending_error) pending_error = maintenance_error_;
    maintenance_stopping_ = true;
  }
  maintenance_queue_cv_.notify_all();
  for (std::thread& worker : maintenance_workers_) {
    if (worker.joinable()) worker.join();
  }
  maintenance_workers_.clear();
  std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
  maintenance_started_ = false;
  if (pending_error) std::rethrow_exception(pending_error);
}

faiss::idx_t IndexIVFFlatLIRE::published_rows() const {
  return published_rows_.load(std::memory_order_acquire);
}

std::vector<LireMaintenanceEvent> IndexIVFFlatLIRE::maintenance_events() const {
  std::lock_guard<std::mutex> lock(maintenance_events_mutex_);
  return maintenance_events_;
}

LireMaintenanceStats IndexIVFFlatLIRE::maintenance_stats() const {
  std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
  LireMaintenanceStats stats;
  stats.jobs_enqueued = maintenance_jobs_enqueued_;
  stats.jobs_started = maintenance_jobs_started_;
  stats.jobs_completed = maintenance_jobs_completed_;
  stats.jobs_retried = maintenance_jobs_retried_;
  stats.pending_jobs = maintenance_queue_.size();
  stats.running_jobs = maintenance_active_workers_;
  stats.peak_pending_jobs = maintenance_peak_pending_jobs_;
  stats.total_queue_wait_ms = maintenance_total_queue_wait_ms_;
  stats.total_execution_ms = maintenance_total_execution_ms_;
  stats.split_jobs_enqueued = split_jobs_enqueued_;
  stats.split_jobs_completed = split_jobs_completed_;
  stats.reassign_jobs_enqueued = reassign_jobs_enqueued_;
  stats.reassign_jobs_started = reassign_jobs_started_;
  stats.reassign_jobs_completed = reassign_jobs_completed_;
  stats.reassign_jobs_dropped_stale = reassign_jobs_dropped_stale_;
  stats.reassign_jobs_noop_same_target = reassign_jobs_noop_same_target_;
  stats.reassign_vectors_moved = reassign_vectors_moved_;
  stats.query_stale_copies_skipped =
      query_stale_copies_skipped_.load(std::memory_order_relaxed);
  stats.maintenance_stale_copies_filtered =
      maintenance_stale_copies_filtered_.load(std::memory_order_relaxed);
  stats.stale_physical_copies_filtered =
      stats.query_stale_copies_skipped +
      stats.maintenance_stale_copies_filtered;
  stats.reassign_source_rescans =
      reassign_source_rescans_.load(std::memory_order_relaxed);
  stats.reassign_payload_bytes_enqueued = reassign_payload_bytes_enqueued_;
  return stats;
}

LireOversizedListStats IndexIVFFlatLIRE::sample_oversized_active_lists(
    bool reset_baseline) const {
  std::vector<uint8_t> current(max_nlist_, 0);
  LireOversizedListStats stats;
  for (uint32_t slot = 0; slot < max_nlist_; ++slot) {
    std::shared_lock<std::shared_mutex> list_lock(list_mutex(slot));
    std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
    if (active_[slot] &&
        invlists->list_size(slot) > config_.split_threshold) {
      current[slot] = 1;
      ++stats.oversized_active_lists;
    }
  }
  std::lock_guard<std::mutex> lock(oversized_stats_mutex_);
  if (reset_baseline || !oversized_stats_initialized_) {
    previous_oversized_lists_ = std::move(current);
    newly_oversized_since_sample_.assign(max_nlist_, 0);
    oversized_stats_initialized_ = true;
    return stats;
  }
  stats.newly_oversized_lists = std::count(
      newly_oversized_since_sample_.begin(),
      newly_oversized_since_sample_.end(), static_cast<uint8_t>(1));
  previous_oversized_lists_ = std::move(current);
  std::fill(newly_oversized_since_sample_.begin(),
            newly_oversized_since_sample_.end(), 0);
  return stats;
}

void IndexIVFFlatLIRE::note_oversized_list_state(
    faiss::idx_t slot, bool active, size_t list_size) {
  if (slot < 0 || slot >= static_cast<faiss::idx_t>(max_nlist_)) return;
  std::lock_guard<std::mutex> lock(oversized_stats_mutex_);
  if (!oversized_stats_initialized_) return;
  const size_t index = static_cast<size_t>(slot);
  const uint8_t now_oversized =
      active && list_size > config_.split_threshold ? 1 : 0;
  if (!previous_oversized_lists_[index] && now_oversized)
    newly_oversized_since_sample_[index] = 1;
  previous_oversized_lists_[index] = now_oversized;
}

void IndexIVFFlatLIRE::reset_maintenance_timeline() {
  wait_for_maintenance();
  maintenance_epoch_ = std::chrono::steady_clock::now();
  maintenance_job_serial_.store(0, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
    maintenance_jobs_enqueued_ = 0;
    maintenance_jobs_started_ = 0;
    maintenance_jobs_completed_ = 0;
    maintenance_jobs_retried_ = 0;
    maintenance_peak_pending_jobs_ = 0;
    maintenance_total_queue_wait_ms_ = 0.0;
    maintenance_total_execution_ms_ = 0.0;
    split_jobs_enqueued_ = 0;
    split_jobs_completed_ = 0;
    reassign_jobs_enqueued_ = 0;
    reassign_jobs_started_ = 0;
    reassign_jobs_completed_ = 0;
    reassign_jobs_dropped_stale_ = 0;
    reassign_jobs_noop_same_target_ = 0;
    reassign_vectors_moved_ = 0;
    reassign_payload_bytes_enqueued_ = 0;
    maintenance_last_log_ = maintenance_epoch_;
  }
  query_stale_copies_skipped_.store(0, std::memory_order_relaxed);
  maintenance_stale_copies_filtered_.store(0, std::memory_order_relaxed);
  reassign_source_rescans_.store(0, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(maintenance_events_mutex_);
  maintenance_events_.clear();
}

void IndexIVFFlatLIRE::validate_consistency() {
  wait_for_maintenance();
  std::vector<std::shared_lock<std::shared_mutex>> list_locks;
  list_locks.reserve(max_nlist_);
  for (uint32_t slot = 0; slot < max_nlist_; ++slot)
    list_locks.emplace_back(list_mutex(slot));
  std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
  validate_state();
}

void IndexIVFFlatLIRE::train(faiss::idx_t n, const float* x) {
  std::lock_guard<std::mutex> insert_lock(insert_mutex_);
  std::unique_lock<std::shared_mutex> lock(state_mutex_);
  if (n <= 0 || x == nullptr) throw std::runtime_error("empty LIRE training set");
  if (ntotal != 0) throw std::runtime_error("cannot retrain non-empty LIRE index");

  // Reuse Faiss IVFFlat's standard global training for the initial nlist. This
  // keeps initial clustering semantics aligned with faiss_ivfflat_run_eval.
  faiss::IndexFlatL2 training_quantizer(d);
  faiss::IndexIVFFlat training_index(
      &training_quantizer, d, initial_nlist_, faiss::METRIC_L2);
  training_index.train(n, x);
  training_quantizer.reconstruct_n(
      0, initial_nlist_, centroids_.data());

  std::fill(active_.begin(), active_.end(), 0);
  free_slots_.clear();
  for (uint32_t slot = 0; slot < initial_nlist_; ++slot) active_[slot] = 1;
  for (uint32_t slot = max_nlist_; slot > initial_nlist_; --slot) {
    free_slots_.push_back(static_cast<faiss::idx_t>(slot - 1));
  }
  active_nlist_ = initial_nlist_;
  topology_generation_ = 0;
  published_rows_.store(0, std::memory_order_release);
  maintenance_epoch_ = std::chrono::steady_clock::now();
  rebuild_quantizer();
  is_trained = true;
}

void IndexIVFFlatLIRE::save_initial_state(const std::string& path,
                                           uint64_t dataset_hash) const {
  std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
  if (!is_trained || ntotal <= 0 || published_rows_.load() != ntotal)
    throw std::runtime_error("cannot save an incomplete LIRE initial index");
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to write LIRE index artifact: " + path);
  out.write("LIREIX01", 8);
  const uint32_t metadata[] = {static_cast<uint32_t>(d), initial_nlist_,
      max_nlist_, active_nlist_};
  out.write(reinterpret_cast<const char*>(metadata), sizeof(metadata));
  const faiss::idx_t rows = ntotal;
  out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char*>(&dataset_hash), sizeof(dataset_hash));
  out.write(reinterpret_cast<const char*>(centroids_.data()),
      static_cast<std::streamsize>(centroids_.size() * sizeof(float)));
  out.write(reinterpret_cast<const char*>(active_.data()),
      static_cast<std::streamsize>(active_.size()));
  for (uint32_t slot = 0; slot < max_nlist_; ++slot) {
    const size_t count = invlists->list_size(slot);
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (!count) continue;
    const faiss::idx_t* ids = invlists->get_ids(slot);
    const uint8_t* codes = invlists->get_codes(slot);
    out.write(reinterpret_cast<const char*>(ids),
        static_cast<std::streamsize>(count * sizeof(faiss::idx_t)));
    out.write(reinterpret_cast<const char*>(codes),
        static_cast<std::streamsize>(count * static_cast<size_t>(d) * sizeof(float)));
    invlists->release_ids(slot, ids);
    invlists->release_codes(slot, codes);
  }
  out.flush();
  if (!out) throw std::runtime_error("failed writing LIRE index artifact: " + path);
}

void IndexIVFFlatLIRE::load_initial_state(const std::string& path,
                                           faiss::idx_t expected_rows,
                                           uint64_t dataset_hash) {
  std::lock_guard<std::mutex> insert_lock(insert_mutex_);
  std::unique_lock<std::shared_mutex> state_lock(state_mutex_);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("unable to open LIRE index artifact: " + path);
  char magic[8]{};
  uint32_t metadata[4]{};
  faiss::idx_t rows = 0;
  uint64_t artifact_dataset_hash = 0;
  if (!in.read(magic, sizeof(magic)) || std::string(magic, 8) != "LIREIX01" ||
      !in.read(reinterpret_cast<char*>(metadata), sizeof(metadata)) ||
      !in.read(reinterpret_cast<char*>(&rows), sizeof(rows)) ||
      !in.read(reinterpret_cast<char*>(&artifact_dataset_hash), sizeof(artifact_dataset_hash)))
    throw std::runtime_error("invalid or truncated LIRE index artifact");
  if (metadata[0] != static_cast<uint32_t>(d) || metadata[1] != initial_nlist_ ||
      metadata[2] != max_nlist_ || metadata[3] < initial_nlist_ ||
      metadata[3] > max_nlist_ || artifact_dataset_hash != dataset_hash ||
      rows != expected_rows || ntotal != 0)
    throw std::runtime_error("LIRE index artifact configuration/row mismatch");
  if (!in.read(reinterpret_cast<char*>(centroids_.data()),
      static_cast<std::streamsize>(centroids_.size() * sizeof(float))) ||
      !in.read(reinterpret_cast<char*>(active_.data()),
      static_cast<std::streamsize>(active_.size())))
    throw std::runtime_error("truncated LIRE index topology");
  std::fill(membership_.begin(), membership_.end(), -1);
  membership_.resize(static_cast<size_t>(rows), -1);
  uint64_t total = 0;
  free_slots_.clear();
  active_nlist_ = 0;
  for (uint32_t slot = 0; slot < max_nlist_; ++slot) {
    if (active_[slot] > 1)
      throw std::runtime_error("invalid active-list marker in LIRE index artifact");
    if (active_[slot]) ++active_nlist_;
    else free_slots_.push_back(static_cast<faiss::idx_t>(slot));
    size_t count = 0;
    if (!in.read(reinterpret_cast<char*>(&count), sizeof(count)) || count > static_cast<size_t>(rows))
      throw std::runtime_error("invalid LIRE index list size");
    std::vector<faiss::idx_t> ids(count);
    std::vector<float> vectors(count * static_cast<size_t>(d));
    if (count && (!in.read(reinterpret_cast<char*>(ids.data()),
          static_cast<std::streamsize>(ids.size() * sizeof(faiss::idx_t))) ||
        !in.read(reinterpret_cast<char*>(vectors.data()),
          static_cast<std::streamsize>(vectors.size() * sizeof(float)))))
      throw std::runtime_error("truncated LIRE index list data");
    for (faiss::idx_t id : ids) {
      if (id < 0 || id >= rows || membership_[static_cast<size_t>(id)] != -1)
        throw std::runtime_error("invalid or duplicate ID in LIRE index artifact");
      membership_[static_cast<size_t>(id)] = slot;
    }
    total += count;
    write_list(slot, ListData{std::move(ids), std::move(vectors)});
  }
  char trailing;
  if (active_nlist_ != metadata[3] || total != static_cast<uint64_t>(rows) ||
      std::find(membership_.begin(), membership_.end(), -1) != membership_.end() ||
      in.read(&trailing, 1))
    throw std::runtime_error("LIRE index artifact payload mismatch");
  ntotal = rows;
  published_rows_.store(rows, std::memory_order_release);
  is_trained = true;
  rebuild_quantizer();
}

void IndexIVFFlatLIRE::add_with_ids(
    faiss::idx_t n,
    const float* x,
    const faiss::idx_t* xids) {
  std::lock_guard<std::mutex> insert_lock(insert_mutex_);
  if (!is_trained) throw std::runtime_error("LIRE index is not trained");
  if (n < 0 || (n > 0 && x == nullptr)) throw std::runtime_error("invalid LIRE add");
  if (n == 0) return;
  std::vector<faiss::idx_t> ids(static_cast<size_t>(n));
  faiss::idx_t begin_id = -1;
  for (;;) {
    uint64_t route_generation = 0;
    std::vector<float> coarse_distances(static_cast<size_t>(n));
    std::vector<faiss::idx_t> assignments(static_cast<size_t>(n));
    {
      std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
      begin_id = ntotal;
      route_generation = topology_generation_;
      for (faiss::idx_t i = 0; i < n; ++i) {
        ids[static_cast<size_t>(i)] = xids ? xids[i] : begin_id + i;
        if (ids[static_cast<size_t>(i)] != begin_id + i) {
          throw std::runtime_error(
              "LIRE run_eval requires sequential IDs equal to insertion order");
        }
      }
#ifdef _OPENMP
      const int previous_omp_threads = omp_get_max_threads();
      omp_set_dynamic(0);
      omp_set_max_active_levels(1);
      omp_set_num_threads(static_cast<int>(std::max<uint32_t>(1, config_.insert_threads)));
#endif
      active_quantizer()->search(n, x, 1, coarse_distances.data(), assignments.data());
#ifdef _OPENMP
      omp_set_num_threads(previous_omp_threads);
#endif
    }

    std::vector<std::vector<size_t>> positions(max_nlist_);
    std::vector<faiss::idx_t> touched;
    for (faiss::idx_t i = 0; i < n; ++i) {
      const faiss::idx_t slot = assignments[static_cast<size_t>(i)];
      if (slot < 0 || slot >= static_cast<faiss::idx_t>(max_nlist_))
        throw std::runtime_error("LIRE quantizer returned an invalid slot");
      if (positions[static_cast<size_t>(slot)].empty()) touched.push_back(slot);
      positions[static_cast<size_t>(slot)].push_back(static_cast<size_t>(i));
    }

    std::sort(touched.begin(), touched.end());
    std::vector<std::unique_lock<std::shared_mutex>> list_locks;
    list_locks.reserve(touched.size());
    for (faiss::idx_t slot : touched) list_locks.emplace_back(list_mutex(slot));
    std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
    if (route_generation != topology_generation_) continue;
    bool assignments_active = true;
    for (faiss::idx_t slot : touched)
      assignments_active = assignments_active && active_[static_cast<size_t>(slot)];
    if (!assignments_active) continue;
    const faiss::idx_t published = published_rows_.load(std::memory_order_relaxed);
    if (published != begin_id)
      throw std::runtime_error("LIRE published row count mismatch");
    std::unique_lock<std::shared_mutex> membership_lock(membership_mutex_);
    membership_.resize(static_cast<size_t>(begin_id + n), -1);
    for (faiss::idx_t slot : touched) {
      const auto& pos = positions[static_cast<size_t>(slot)];
      std::vector<faiss::idx_t> list_ids(pos.size());
      std::vector<float> list_vectors(pos.size() * static_cast<size_t>(d));
      for (size_t j = 0; j < pos.size(); ++j) {
        const size_t input_index = pos[j];
        list_ids[j] = ids[input_index];
        std::copy(x + input_index * static_cast<size_t>(d),
                  x + (input_index + 1) * static_cast<size_t>(d),
                  list_vectors.data() + j * static_cast<size_t>(d));
      }
      invlists->add_entries(
          static_cast<size_t>(slot), list_ids.size(), list_ids.data(),
          reinterpret_cast<const uint8_t*>(list_vectors.data()));
      note_oversized_list_state(
          slot, true, invlists->list_size(static_cast<size_t>(slot)));
      for (size_t position : pos)
        membership_[static_cast<size_t>(ids[position])] = slot;
    }
    published_rows_.store(begin_id + n, std::memory_order_release);
    membership_lock.unlock();
    state_lock.unlock();
    list_locks.clear();
    {
      std::unique_lock<std::shared_mutex> metadata_lock(state_mutex_);
      ntotal = begin_id + n;
    }
    if (config_.async_maintenance) {
      for (faiss::idx_t slot : touched) enqueue_maintenance(slot);
    } else {
      run_maintenance(touched);
    }
    break;
  }
  if (config_.validate_invariants && !config_.async_maintenance) {
    validate_consistency();
  }
}

void IndexIVFFlatLIRE::search(
    faiss::idx_t n,
    const float* x,
    faiss::idx_t k,
    float* distances,
    faiss::idx_t* labels,
    const faiss::SearchParameters* params) const {
  (void)params;
  search_at_frontier(n, x, k, distances, labels, published_rows());
}

void IndexIVFFlatLIRE::search_at_frontier(
    faiss::idx_t n,
    const float* x,
    faiss::idx_t k,
    float* distances,
    faiss::idx_t* labels,
    faiss::idx_t visible_rows,
    QuerySearchTeam* query_team) const {
  if (n < 0 || k < 0 || visible_rows < 0 || (n > 0 && x == nullptr))
    throw std::runtime_error("invalid LIRE search");
  if (query_team && n != 1)
    throw std::runtime_error("per-request query team requires exactly one query");
  if (n == 0 || k == 0) return;
  const float infinity = std::numeric_limits<float>::infinity();
  std::fill(distances, distances + static_cast<size_t>(n) * k, infinity);
  std::fill(labels, labels + static_cast<size_t>(n) * k, -1);

  std::vector<std::vector<faiss::idx_t>> routes(static_cast<size_t>(n));
  std::vector<faiss::idx_t> locked_slots;
  std::vector<std::shared_lock<std::shared_mutex>> list_locks;
  // Route and posting state must describe the same topology. A split may
  // publish after routing but before posting locks are acquired, so retry the
  // inexpensive route step if the topology changed in that gap.
  for (;;) {
    for (auto& route : routes) route.clear();
    locked_slots.clear();
    uint64_t route_generation = 0;
    {
      std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
      route_generation = topology_generation_;
      const uint32_t probe_count = std::min<uint32_t>(nprobe, active_nlist_);
      for (faiss::idx_t qi = 0; qi < n; ++qi) {
        std::vector<std::pair<float, faiss::idx_t>> ranked;
        ranked.reserve(active_nlist_);
        for (uint32_t slot = 0; slot < max_nlist_; ++slot) {
          if (!active_[slot]) continue;
          ranked.emplace_back(l2(x + static_cast<size_t>(qi) * d, centroid(slot)),
                              static_cast<faiss::idx_t>(slot));
        }
        const size_t keep = std::min<size_t>(probe_count, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + keep, ranked.end(),
                          [](const auto& a, const auto& b) {
                            return a.first < b.first ||
                                   (a.first == b.first && a.second < b.second);
                          });
        for (size_t j = 0; j < keep; ++j) {
          routes[static_cast<size_t>(qi)].push_back(ranked[j].second);
          locked_slots.push_back(ranked[j].second);
        }
      }
    }
    std::sort(locked_slots.begin(), locked_slots.end());
    locked_slots.erase(std::unique(locked_slots.begin(), locked_slots.end()),
                       locked_slots.end());
    list_locks.clear();
    list_locks.reserve(locked_slots.size());
    for (faiss::idx_t slot : locked_slots)
      list_locks.emplace_back(list_mutex(slot));
    bool topology_matches = false;
    {
      std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
      topology_matches = route_generation == topology_generation_;
    }
    if (topology_matches) break;
    list_locks.clear();
  }
  // Acquiring membership only after every posting lock avoids lock inversion
  // with per-vector publication (posting locks -> membership lock).
  std::shared_lock<std::shared_mutex> membership_lock(membership_mutex_);

  const auto candidate_order = [](const auto& a, const auto& b) {
    return a.first < b.first ||
           (a.first == b.first && a.second < b.second);
  };
  for (faiss::idx_t qi = 0; qi < n; ++qi) {
    std::vector<std::pair<float, faiss::idx_t>> best;
    const float* query = x + static_cast<size_t>(qi) * d;
    auto scan_list = [&](std::vector<std::pair<float, faiss::idx_t>>& output,
                         faiss::idx_t slot) {
      const size_t list_size = invlists->list_size(static_cast<size_t>(slot));
      if (list_size == 0) return;
      const faiss::idx_t* ids = invlists->get_ids(static_cast<size_t>(slot));
      const uint8_t* codes = invlists->get_codes(static_cast<size_t>(slot));
      const float* vectors = reinterpret_cast<const float*>(codes);
      try {
        for (size_t offset = 0; offset < list_size; ++offset) {
          const faiss::idx_t id = ids[offset];
          if (id < 0 || id >= visible_rows) continue;
          if (static_cast<size_t>(id) >= membership_.size() ||
              membership_[static_cast<size_t>(id)] != slot) {
            query_stale_copies_skipped_.fetch_add(
                1, std::memory_order_relaxed);
            continue;
          }
          const float distance = l2(query, vectors + offset * static_cast<size_t>(d));
          output.emplace_back(distance, id);
        }
      } catch (...) {
        invlists->release_ids(static_cast<size_t>(slot), ids);
        invlists->release_codes(static_cast<size_t>(slot), codes);
        throw;
      }
      invlists->release_ids(static_cast<size_t>(slot), ids);
      invlists->release_codes(static_cast<size_t>(slot), codes);
    };
    if (query_team) {
      std::vector<std::vector<std::pair<float, faiss::idx_t>>> local_best(
          query_team->thread_count());
      query_team->parallel_for(routes[static_cast<size_t>(qi)].size(),
          [&](uint32_t worker, size_t route_index) {
            scan_list(local_best[worker], routes[static_cast<size_t>(qi)][route_index]);
          });
      for (auto& local : local_best) {
        const size_t local_keep = std::min<size_t>(static_cast<size_t>(k), local.size());
        if (local.size() > local_keep) {
          std::partial_sort(local.begin(), local.begin() + local_keep,
                            local.end(), candidate_order);
          local.resize(local_keep);
        }
        best.insert(best.end(), local.begin(), local.end());
      }
    } else {
      for (faiss::idx_t slot : routes[static_cast<size_t>(qi)])
        scan_list(best, slot);
    }
    const size_t keep = std::min<size_t>(static_cast<size_t>(k), best.size());
    std::partial_sort(best.begin(), best.begin() + keep, best.end(), candidate_order);
    for (size_t j = 0; j < keep; ++j) {
      distances[static_cast<size_t>(qi) * k + j] = best[j].first;
      labels[static_cast<size_t>(qi) * k + j] = best[j].second;
    }
  }
}

void IndexIVFFlatLIRE::reset() {
  std::lock_guard<std::mutex> insert_lock(insert_mutex_);
  wait_for_maintenance();
  std::vector<std::unique_lock<std::shared_mutex>> list_locks;
  list_locks.reserve(max_nlist_);
  for (uint32_t slot = 0; slot < max_nlist_; ++slot)
    list_locks.emplace_back(list_mutex(static_cast<faiss::idx_t>(slot)));
  std::unique_lock<std::shared_mutex> state_lock(state_mutex_);
  for (uint32_t slot = 0; slot < max_nlist_; ++slot) {
    invlists->resize(slot, 0);
    note_oversized_list_state(slot, active_[slot], 0);
  }
  {
    std::unique_lock<std::shared_mutex> membership_lock(membership_mutex_);
    membership_.clear();
  }
  ntotal = 0;
  published_rows_.store(0, std::memory_order_release);
  split_count_ = 0;
  reassign_scanned_.store(0, std::memory_order_relaxed);
  reassign_moved_.store(0, std::memory_order_relaxed);
}

size_t IndexIVFFlatLIRE::remove_ids(const faiss::IDSelector&) {
  throw std::runtime_error("IndexIVFFlatLIRE does not support deletion");
}

void IndexIVFFlatLIRE::validate_state() const {
  std::shared_lock<std::shared_mutex> membership_lock(membership_mutex_);
  if (membership_.size() != static_cast<size_t>(ntotal)) {
    throw std::runtime_error("LIRE membership size mismatch");
  }
  std::vector<uint8_t> seen(static_cast<size_t>(ntotal), 0);
  size_t logical_total = 0;
  uint32_t active_count = 0;
  for (uint32_t slot = 0; slot < max_nlist_; ++slot) {
    const size_t size = invlists->list_size(slot);
    if (active_[slot]) {
      active_count++;
    } else if (size != 0) {
      throw std::runtime_error("inactive LIRE list is not empty");
    }
    if (size == 0) continue;
    const faiss::idx_t* ids = invlists->get_ids(slot);
    for (size_t offset = 0; offset < size; ++offset) {
      const faiss::idx_t id = ids[offset];
      if (id < 0 || id >= ntotal) {
        invlists->release_ids(slot, ids);
        throw std::runtime_error("LIRE list contains out-of-range ID");
      }
      if (membership_[static_cast<size_t>(id)] ==
          static_cast<faiss::idx_t>(slot)) {
        if (seen[static_cast<size_t>(id)]++) {
          invlists->release_ids(slot, ids);
          throw std::runtime_error("LIRE has duplicate authoritative copies");
        }
        ++logical_total;
      }
    }
    invlists->release_ids(slot, ids);
  }
  if (logical_total != static_cast<size_t>(ntotal) ||
      active_count != active_nlist_ ||
      std::find(seen.begin(), seen.end(), static_cast<uint8_t>(0)) != seen.end()) {
    throw std::runtime_error("LIRE index cardinality mismatch");
  }
}

uint32_t IndexIVFFlatLIRE::active_nlist() const {
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  return active_nlist_;
}

uint64_t IndexIVFFlatLIRE::split_count() const {
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  return split_count_;
}

uint64_t IndexIVFFlatLIRE::reassign_scanned() const {
  return reassign_scanned_.load(std::memory_order_relaxed);
}

uint64_t IndexIVFFlatLIRE::reassign_moved() const {
  return reassign_moved_.load(std::memory_order_relaxed);
}

}  // namespace baseline_eval
