#include "faiss_ivfflat_lire.h"

#include <faiss/IndexFlat.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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
      list_generation_(config.max_nlist, 0),
      maintenance_queued_or_running_(config.max_nlist, 0),
      maintenance_request_ms_(config.max_nlist, 0.0) {
  if (d <= 0 || initial_nlist_ == 0 || max_nlist_ < initial_nlist_ ||
      config_.split_threshold < 2) {
    throw std::runtime_error("invalid IndexIVFFlatLIRE configuration");
  }
  if (config_.async_maintenance && config_.maintenance_threads != 1) {
    throw std::runtime_error(
        "async IndexIVFFlatLIRE currently requires exactly one maintenance worker");
  }
  own_fields = true;
  is_trained = false;
  list_mutexes_.reserve(max_nlist_);
  for (uint32_t i = 0; i < max_nlist_; ++i) {
    list_mutexes_.push_back(std::make_unique<std::shared_mutex>());
  }
  maintenance_epoch_ = std::chrono::steady_clock::now();
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
    uint64_t* reassigned_vectors) {
  *did_split = false;
  *reassigned_vectors = 0;
  ListData input;
  std::vector<float> old_center;
  std::vector<faiss::idx_t> neighbors;
  uint64_t source_generation = 0;
  {
    std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
    if (!active_[static_cast<size_t>(slot)]) return {};
    std::shared_lock<std::shared_mutex> list_lock(list_mutex(slot));
    if (invlists->list_size(static_cast<size_t>(slot)) <=
        config_.split_threshold) return {};
    input = read_list(slot);
    source_generation = list_generation_[static_cast<size_t>(slot)];
    old_center.assign(centroid(slot), centroid(slot) + d);
    neighbors = nearest_centroid_slots(old_center.data(), slot,
                                       config_.reassign_k);
  }

  std::vector<uint8_t> labels;
  std::vector<float> new_centers;
  if (!split_balanced_kmeans(input, &labels, &new_centers)) return {};

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

  faiss::idx_t new_slot = -1;
  {
    // Follow SPFresh's per-posting synchronization model. The topology lock is
    // held only for the short list/centroid publication, never for clustering.
    std::unique_lock<std::shared_mutex> state_lock(state_mutex_);
    if (!active_[static_cast<size_t>(slot)] ||
        list_generation_[static_cast<size_t>(slot)] != source_generation) {
      return {slot};
    }
    if (free_slots_.empty()) {
      throw std::runtime_error("LIRE max_nlist exhausted during split");
    }
    new_slot = free_slots_.back();
    std::vector<faiss::idx_t> lock_slots{slot, new_slot};
    std::sort(lock_slots.begin(), lock_slots.end());
    std::vector<std::unique_lock<std::shared_mutex>> list_locks;
    list_locks.reserve(lock_slots.size());
    for (faiss::idx_t list : lock_slots) {
      list_locks.emplace_back(list_mutex(list));
    }
    if (list_generation_[static_cast<size_t>(slot)] != source_generation) {
      return {slot};
    }
    free_slots_.pop_back();
    write_list(slot, children[0]);
    write_list(new_slot, children[1]);
    std::copy(new_centers.begin(), new_centers.begin() + d, centroid(slot));
    std::copy(new_centers.begin() + d, new_centers.end(), centroid(new_slot));
    active_[static_cast<size_t>(new_slot)] = 1;
    active_nlist_++;
    {
      std::lock_guard<std::mutex> membership_lock(membership_mutex_);
      for (faiss::idx_t id : children[0].ids)
        membership_[static_cast<size_t>(id)] = slot;
      for (faiss::idx_t id : children[1].ids)
        membership_[static_cast<size_t>(id)] = new_slot;
    }
    ++list_generation_[static_cast<size_t>(slot)];
    ++list_generation_[static_cast<size_t>(new_slot)];
    ++split_count_;
    rebuild_quantizer();
  }
  *did_split = true;

  std::vector<faiss::idx_t> changed{slot, new_slot};
  if (config_.enable_reassign) {
    std::vector<faiss::idx_t> moved_targets = collect_and_reassign(
        slot, old_center, slot, new_slot, neighbors, reassigned_vectors);
    changed.insert(changed.end(), moved_targets.begin(), moved_targets.end());
  }
  return changed;
}

std::vector<faiss::idx_t> IndexIVFFlatLIRE::collect_and_reassign(
    faiss::idx_t old_slot,
    const std::vector<float>& old_center,
    faiss::idx_t child0,
    faiss::idx_t child1,
    const std::vector<faiss::idx_t>& neighbors,
    uint64_t* reassigned_vectors) {
  (void)old_slot;
  struct Candidate {
    faiss::idx_t id;
    faiss::idx_t source;
    std::vector<float> vector;
  };

  std::vector<Candidate> candidate_vectors;
  std::unordered_set<faiss::idx_t> candidate_ids;
  std::vector<float> centroid_snapshot;
  std::vector<uint8_t> active_snapshot;
  std::unordered_map<faiss::idx_t, ListData> list_snapshots;
  {
    std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
    centroid_snapshot = centroids_;
    active_snapshot = active_;
    std::vector<faiss::idx_t> slots{child0, child1};
    for (faiss::idx_t neighbor : neighbors) {
      if (neighbor >= 0 && neighbor < static_cast<faiss::idx_t>(max_nlist_) &&
          active_[static_cast<size_t>(neighbor)]) {
        slots.push_back(neighbor);
      }
    }
    std::sort(slots.begin(), slots.end());
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
    std::vector<std::shared_lock<std::shared_mutex>> list_locks;
    list_locks.reserve(slots.size());
    for (faiss::idx_t list : slots) list_locks.emplace_back(list_mutex(list));
    for (faiss::idx_t list : slots) list_snapshots.emplace(list, read_list(list));
  }
  const auto center_for = [&](faiss::idx_t list) {
    return centroid_snapshot.data() + static_cast<size_t>(list) * d;
  };
  const auto nearest_snapshot = [&](const float* vector) {
    faiss::idx_t best = -1;
    float best_distance = std::numeric_limits<float>::infinity();
    for (uint32_t candidate = 0; candidate < max_nlist_; ++candidate) {
      if (!active_snapshot[candidate]) continue;
      const float distance = l2(vector, center_for(candidate));
      if (distance < best_distance) {
        best_distance = distance;
        best = static_cast<faiss::idx_t>(candidate);
      }
    }
    return best;
  };
  auto add_candidate = [&](faiss::idx_t id,
                           faiss::idx_t source,
                           const float* vector) {
    if (!candidate_ids.insert(id).second) return;
    Candidate candidate;
    candidate.id = id;
    candidate.source = source;
    candidate.vector.assign(vector, vector + d);
    candidate_vectors.push_back(std::move(candidate));
  };

  auto check_child = [&](faiss::idx_t child) {
    const ListData& list = list_snapshots.at(child);
    for (size_t i = 0; i < list.size(); ++i) {
      const float* vector = list.row(i, d);
      reassign_scanned_++;
      // SPFresh CheckIsNeedReassign(..., isInSplitHead=true):
      // reject when D(v, old) >= D(v, current new head).
      if (l2(vector, old_center.data()) >= l2(vector, center_for(child))) continue;
      add_candidate(list.ids[i], child, vector);
    }
  };
  check_child(child0);
  check_child(child1);

  for (faiss::idx_t neighbor : neighbors) {
    if (!active_snapshot[static_cast<size_t>(neighbor)] || neighbor == child0 ||
        neighbor == child1) {
      continue;
    }
    const ListData& list = list_snapshots.at(neighbor);
    for (size_t i = 0; i < list.size(); ++i) {
      const float* vector = list.row(i, d);
      reassign_scanned_++;
      const float old_distance = l2(vector, old_center.data());
      const float new0 = l2(vector, center_for(child0));
      const float new1 = l2(vector, center_for(child1));
      const float current = l2(vector, center_for(neighbor));
      // SPFresh CheckIsNeedReassign(..., isInSplitHead=false).
      if (old_distance <= new0 && old_distance <= new1) continue;
      if (current <= new0 && current <= new1) continue;
      add_candidate(list.ids[i], neighbor, vector);
    }
  }

  std::vector<Move> moves;
  moves.reserve(candidate_vectors.size());
  for (const Candidate& candidate : candidate_vectors) {
    const faiss::idx_t target = nearest_snapshot(candidate.vector.data());
    if (target == candidate.source) continue;
    Move move;
    move.id = candidate.id;
    move.source = candidate.source;
    move.target = target;
    move.vector = candidate.vector;
    moves.push_back(std::move(move));
  }

  *reassigned_vectors = apply_moves(moves);
  std::vector<faiss::idx_t> targets;
  targets.reserve(moves.size());
  for (const Move& move : moves) targets.push_back(move.target);
  std::sort(targets.begin(), targets.end());
  targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
  return targets;
}

uint64_t IndexIVFFlatLIRE::apply_moves(const std::vector<Move>& moves) {
  if (moves.empty()) return 0;
  std::unordered_map<faiss::idx_t, ListData> lists;
  std::unordered_map<faiss::idx_t, faiss::idx_t> target_by_id;
  for (const Move& move : moves) {
    if (!lists.count(move.source)) lists.emplace(move.source, ListData{});
    if (!lists.count(move.target)) lists.emplace(move.target, ListData{});
    target_by_id.emplace(move.id, move.target);
  }
  // Conservative first-version publication: the global state lock protects
  // membership/list consistency while affected lists are materialized and
  // rewritten. This can block query routing during reassignment commit and is
  // a later tuning point.
  std::unique_lock<std::shared_mutex> state_lock(state_mutex_);
  std::vector<faiss::idx_t> slots;
  slots.reserve(lists.size());
  for (const auto& entry : lists) slots.push_back(entry.first);
  std::sort(slots.begin(), slots.end());
  std::vector<std::unique_lock<std::shared_mutex>> list_locks;
  list_locks.reserve(slots.size());
  for (faiss::idx_t slot : slots) list_locks.emplace_back(list_mutex(slot));
  for (faiss::idx_t slot : slots) lists.at(slot) = read_list(slot);
  std::lock_guard<std::mutex> membership_lock(membership_mutex_);
  std::vector<Move> valid_moves;
  valid_moves.reserve(moves.size());
  for (const Move& move : moves) {
    if (move.id >= 0 && static_cast<size_t>(move.id) < membership_.size() &&
        membership_[static_cast<size_t>(move.id)] == move.source &&
        active_[static_cast<size_t>(move.target)]) {
      valid_moves.push_back(move);
    }
  }
  target_by_id.clear();
  for (const Move& move : valid_moves) target_by_id.emplace(move.id, move.target);
  for (auto& entry : lists) {
    ListData filtered;
    filtered.ids.reserve(entry.second.size());
    filtered.vectors.reserve(entry.second.vectors.size());
    for (size_t i = 0; i < entry.second.size(); ++i) {
      const faiss::idx_t id = entry.second.ids[i];
      auto move_it = target_by_id.find(id);
      if (move_it != target_by_id.end() &&
          membership_[static_cast<size_t>(id)] == entry.first) {
        continue;
      }
      filtered.ids.push_back(id);
      const float* vector = entry.second.row(i, d);
      filtered.vectors.insert(filtered.vectors.end(), vector, vector + d);
    }
    entry.second = std::move(filtered);
  }

  for (const Move& move : valid_moves) {
    ListData& target = lists.at(move.target);
    target.ids.push_back(move.id);
    target.vectors.insert(target.vectors.end(), move.vector.begin(), move.vector.end());
  }
  for (const auto& entry : lists) {
    write_list(entry.first, entry.second);
    ++list_generation_[static_cast<size_t>(entry.first)];
  }
  for (const Move& move : valid_moves)
    membership_[static_cast<size_t>(move.id)] = move.target;
  reassign_moved_ += valid_moves.size();
  return valid_moves.size();
}

IndexIVFFlatLIRE::MaintenanceResult IndexIVFFlatLIRE::run_maintenance(
    const std::vector<faiss::idx_t>& touched) {
  MaintenanceResult result;
  std::deque<faiss::idx_t> queue;
  std::vector<uint8_t> queued(max_nlist_, 0);
  for (faiss::idx_t slot : touched) {
    if (slot < 0 || slot >= static_cast<faiss::idx_t>(max_nlist_)) continue;
    std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
    std::shared_lock<std::shared_mutex> list_lock(list_mutex(slot));
    if (!queued[static_cast<size_t>(slot)] && active_[static_cast<size_t>(slot)] &&
        invlists->list_size(static_cast<size_t>(slot)) > config_.split_threshold) {
      queue.push_back(slot);
      queued[static_cast<size_t>(slot)] = 1;
    }
  }

  uint32_t splits = 0;
  while (!queue.empty()) {
    const faiss::idx_t slot = queue.front();
    queue.pop_front();
    queued[static_cast<size_t>(slot)] = 0;
    {
      std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
      if (!active_[static_cast<size_t>(slot)]) continue;
      std::shared_lock<std::shared_mutex> list_lock(list_mutex(slot));
      if (invlists->list_size(static_cast<size_t>(slot)) <=
          config_.split_threshold) continue;
    }
    if (++splits > config_.max_splits_per_batch) {
      throw std::runtime_error("LIRE exceeded max_splits_per_batch");
    }
    bool did_split = false;
    uint64_t moved = 0;
    const std::vector<faiss::idx_t> changed = split_one(slot, &did_split, &moved);
    result.split_count += did_split ? 1u : 0u;
    result.reassigned_vectors += moved;
    result.affected_lists.insert(result.affected_lists.end(), changed.begin(), changed.end());
    for (faiss::idx_t changed_slot : changed) {
      std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
      if (!active_[static_cast<size_t>(changed_slot)]) continue;
      std::shared_lock<std::shared_mutex> list_lock(list_mutex(changed_slot));
      if (!queued[static_cast<size_t>(changed_slot)] &&
          invlists->list_size(static_cast<size_t>(changed_slot)) > config_.split_threshold) {
        queue.push_back(changed_slot);
        queued[static_cast<size_t>(changed_slot)] = 1;
      }
    }
  }
  std::sort(result.affected_lists.begin(), result.affected_lists.end());
  result.affected_lists.erase(
      std::unique(result.affected_lists.begin(), result.affected_lists.end()),
      result.affected_lists.end());
  return result;
}

void IndexIVFFlatLIRE::enqueue_maintenance(faiss::idx_t slot) {
  if (slot < 0 || slot >= static_cast<faiss::idx_t>(max_nlist_)) return;
  std::lock_guard<std::mutex> queue_lock(maintenance_queue_mutex_);
  if (maintenance_stopping_) return;
  {
    std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
    std::shared_lock<std::shared_mutex> list_lock(list_mutex(slot));
    if (!active_[static_cast<size_t>(slot)] ||
        invlists->list_size(static_cast<size_t>(slot)) <= config_.split_threshold) {
      return;
    }
  }
  if (maintenance_queued_or_running_[static_cast<size_t>(slot)]) return;
  maintenance_queued_or_running_[static_cast<size_t>(slot)] = 1;
  maintenance_request_ms_[static_cast<size_t>(slot)] =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - maintenance_epoch_).count();
  maintenance_queue_.push_back(slot);
  maintenance_queue_cv_.notify_one();
}

void IndexIVFFlatLIRE::maintenance_worker() {
  for (;;) {
    faiss::idx_t slot = -1;
    {
      std::unique_lock<std::mutex> lock(maintenance_queue_mutex_);
      maintenance_queue_cv_.wait(lock, [&] {
        return maintenance_stopping_ || !maintenance_queue_.empty();
      });
      if (maintenance_stopping_ && maintenance_queue_.empty()) return;
      slot = maintenance_queue_.front();
      maintenance_queue_.pop_front();
      ++maintenance_active_workers_;
    }

    LireMaintenanceEvent event;
    event.triggering_list = slot;
    bool completed = false;
    uint64_t generation_before_job = 0;
    {
      std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
      event.request_ms = maintenance_request_ms_[static_cast<size_t>(slot)];
    }
    try {
      {
        std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
        std::shared_lock<std::shared_mutex> list_lock(list_mutex(slot));
        generation_before_job = list_generation_[static_cast<size_t>(slot)];
      }
      event.job_id = maintenance_job_serial_.fetch_add(1, std::memory_order_relaxed) + 1;
      event.start_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - maintenance_epoch_).count();
      const MaintenanceResult result = run_maintenance({slot});
      event.compute_done_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - maintenance_epoch_).count();
      event.commit_done_ms = event.compute_done_ms;
      event.end_ms = event.compute_done_ms;
      event.split_count = result.split_count;
      event.reassigned_vectors = result.reassigned_vectors;
      event.affected_lists = static_cast<uint32_t>(result.affected_lists.size());
      event.affected_list_ids = result.affected_lists;
      {
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
      maintenance_queued_or_running_[static_cast<size_t>(slot)] = 0;
    }
    if (completed) {
      bool changed_during_completion = false;
      {
        std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
        std::shared_lock<std::shared_mutex> list_lock(list_mutex(slot));
        changed_during_completion =
            list_generation_[static_cast<size_t>(slot)] != generation_before_job;
      }
      if (changed_during_completion) enqueue_maintenance(slot);
    }
    {
      std::lock_guard<std::mutex> lock(maintenance_queue_mutex_);
      --maintenance_active_workers_;
      if (maintenance_queue_.empty() && maintenance_active_workers_ == 0)
        maintenance_idle_cv_.notify_all();
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

void IndexIVFFlatLIRE::reset_maintenance_timeline() {
  wait_for_maintenance();
  maintenance_epoch_ = std::chrono::steady_clock::now();
  maintenance_job_serial_.store(0, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(maintenance_events_mutex_);
  maintenance_events_.clear();
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
  std::fill(list_generation_.begin(), list_generation_.end(), 0);
  published_rows_.store(0, std::memory_order_release);
  maintenance_epoch_ = std::chrono::steady_clock::now();
  rebuild_quantizer();
  is_trained = true;
}

void IndexIVFFlatLIRE::add_with_ids(
    faiss::idx_t n,
    const float* x,
    const faiss::idx_t* xids) {
  std::lock_guard<std::mutex> insert_lock(insert_mutex_);
  if (!is_trained) throw std::runtime_error("LIRE index is not trained");
  if (n < 0 || (n > 0 && x == nullptr)) throw std::runtime_error("invalid LIRE add");
  if (n == 0) return;

  std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
  std::vector<faiss::idx_t> ids(static_cast<size_t>(n));
  for (faiss::idx_t i = 0; i < n; ++i) {
    ids[static_cast<size_t>(i)] = xids ? xids[i] : ntotal + i;
    if (ids[static_cast<size_t>(i)] != ntotal + i) {
      throw std::runtime_error(
          "LIRE run_eval requires sequential IDs equal to insertion order");
    }
  }

  std::vector<float> coarse_distances(static_cast<size_t>(n));
  std::vector<faiss::idx_t> assignments(static_cast<size_t>(n));
#ifdef _OPENMP
  const int previous_omp_threads = omp_get_max_threads();
  omp_set_num_threads(static_cast<int>(std::max<uint32_t>(1, config_.insert_threads)));
#endif
  active_quantizer()->search(n, x, 1, coarse_distances.data(), assignments.data());
#ifdef _OPENMP
  omp_set_num_threads(previous_omp_threads);
#endif

  std::vector<std::vector<size_t>> positions(max_nlist_);
  std::vector<faiss::idx_t> touched;
  for (faiss::idx_t i = 0; i < n; ++i) {
    const faiss::idx_t slot = assignments[static_cast<size_t>(i)];
    if (slot < 0 || slot >= static_cast<faiss::idx_t>(max_nlist_) ||
        !active_[static_cast<size_t>(slot)]) {
      throw std::runtime_error("LIRE quantizer returned an inactive slot");
    }
    if (positions[static_cast<size_t>(slot)].empty()) touched.push_back(slot);
    positions[static_cast<size_t>(slot)].push_back(static_cast<size_t>(i));
  }

  std::vector<std::unique_lock<std::shared_mutex>> list_locks;
  std::sort(touched.begin(), touched.end());
  list_locks.reserve(touched.size());
  for (faiss::idx_t slot : touched) list_locks.emplace_back(list_mutex(slot));
  const faiss::idx_t begin_id = published_rows_.load(std::memory_order_relaxed);
  if (begin_id != ntotal) throw std::runtime_error("LIRE published row count mismatch");
  {
    std::lock_guard<std::mutex> membership_lock(membership_mutex_);
    membership_.resize(static_cast<size_t>(begin_id + n), -1);
  }
  for (faiss::idx_t slot : touched) {
    const auto& pos = positions[static_cast<size_t>(slot)];
    std::vector<faiss::idx_t> list_ids(pos.size());
    std::vector<float> list_vectors(pos.size() * static_cast<size_t>(d));
    for (size_t j = 0; j < pos.size(); ++j) {
      const size_t input_index = pos[j];
      const faiss::idx_t id = ids[input_index];
      list_ids[j] = id;
      std::copy(x + input_index * static_cast<size_t>(d),
                x + (input_index + 1) * static_cast<size_t>(d),
                list_vectors.data() + j * static_cast<size_t>(d));
    }
    invlists->add_entries(
        static_cast<size_t>(slot),
        list_ids.size(),
        list_ids.data(),
        reinterpret_cast<const uint8_t*>(list_vectors.data()));
    ++list_generation_[static_cast<size_t>(slot)];
  }
  {
    std::lock_guard<std::mutex> membership_lock(membership_mutex_);
    for (faiss::idx_t slot : touched) {
      for (size_t position : positions[static_cast<size_t>(slot)])
        membership_[static_cast<size_t>(ids[position])] = slot;
    }
  }
  published_rows_.store(begin_id + n, std::memory_order_release);
  list_locks.clear();
  state_lock.unlock();
  {
    std::unique_lock<std::shared_mutex> metadata_lock(state_mutex_);
    ntotal = begin_id + n;
  }

  if (config_.async_maintenance) {
    for (faiss::idx_t slot : touched) enqueue_maintenance(slot);
  } else {
    run_maintenance(touched);
  }
  if (config_.validate_invariants) {
    wait_for_maintenance();
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    validate_state();
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
    faiss::idx_t visible_rows) const {
  if (n < 0 || k < 0 || visible_rows < 0 || (n > 0 && x == nullptr))
    throw std::runtime_error("invalid LIRE search");
  if (n == 0 || k == 0) return;
  const float infinity = std::numeric_limits<float>::infinity();
  std::fill(distances, distances + static_cast<size_t>(n) * k, infinity);
  std::fill(labels, labels + static_cast<size_t>(n) * k, -1);

  std::vector<std::vector<faiss::idx_t>> routes(static_cast<size_t>(n));
  std::vector<faiss::idx_t> locked_slots;
  std::vector<std::shared_lock<std::shared_mutex>> list_locks;
  std::shared_lock<std::shared_mutex> state_lock(state_mutex_);
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
  std::sort(locked_slots.begin(), locked_slots.end());
  locked_slots.erase(std::unique(locked_slots.begin(), locked_slots.end()),
                     locked_slots.end());
  list_locks.reserve(locked_slots.size());
  for (faiss::idx_t slot : locked_slots)
    list_locks.emplace_back(list_mutex(slot));
  state_lock.unlock();

  for (faiss::idx_t qi = 0; qi < n; ++qi) {
    std::vector<std::pair<float, faiss::idx_t>> best;
    const float* query = x + static_cast<size_t>(qi) * d;
    for (faiss::idx_t slot : routes[static_cast<size_t>(qi)]) {
      const size_t list_size = invlists->list_size(static_cast<size_t>(slot));
      if (list_size == 0) continue;
      const faiss::idx_t* ids = invlists->get_ids(static_cast<size_t>(slot));
      const uint8_t* codes = invlists->get_codes(static_cast<size_t>(slot));
      const float* vectors = reinterpret_cast<const float*>(codes);
      for (size_t offset = 0; offset < list_size; ++offset) {
        if (ids[offset] < 0 || ids[offset] >= visible_rows) continue;
        const float distance = l2(query, vectors + offset * static_cast<size_t>(d));
        best.emplace_back(distance, ids[offset]);
      }
      invlists->release_ids(static_cast<size_t>(slot), ids);
      invlists->release_codes(static_cast<size_t>(slot), codes);
    }
    const size_t keep = std::min<size_t>(static_cast<size_t>(k), best.size());
    std::partial_sort(best.begin(), best.begin() + keep, best.end(),
                      [](const auto& a, const auto& b) {
                        return a.first < b.first ||
                               (a.first == b.first && a.second < b.second);
                      });
    for (size_t j = 0; j < keep; ++j) {
      distances[static_cast<size_t>(qi) * k + j] = best[j].first;
      labels[static_cast<size_t>(qi) * k + j] = best[j].second;
    }
  }
}

void IndexIVFFlatLIRE::reset() {
  wait_for_maintenance();
  std::lock_guard<std::mutex> insert_lock(insert_mutex_);
  std::unique_lock<std::shared_mutex> lock(state_mutex_);
  for (uint32_t slot = 0; slot < max_nlist_; ++slot) {
    std::unique_lock<std::shared_mutex> list_lock(list_mutex(slot));
    invlists->resize(slot, 0);
    ++list_generation_[slot];
  }
  {
    std::lock_guard<std::mutex> membership_lock(membership_mutex_);
    membership_.clear();
  }
  ntotal = 0;
  published_rows_.store(0, std::memory_order_release);
  split_count_ = 0;
  reassign_scanned_.store(0, std::memory_order_relaxed);
  reassign_moved_ = 0;
}

size_t IndexIVFFlatLIRE::remove_ids(const faiss::IDSelector&) {
  throw std::runtime_error("IndexIVFFlatLIRE does not support deletion");
}

void IndexIVFFlatLIRE::validate_state() const {
  std::lock_guard<std::mutex> membership_lock(membership_mutex_);
  if (membership_.size() != static_cast<size_t>(ntotal)) {
    throw std::runtime_error("LIRE membership size mismatch");
  }
  std::vector<uint8_t> seen(static_cast<size_t>(ntotal), 0);
  size_t total = 0;
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
      if (seen[static_cast<size_t>(id)]++) {
        invlists->release_ids(slot, ids);
        throw std::runtime_error("LIRE list contains duplicate ID");
      }
      if (membership_[static_cast<size_t>(id)] !=
          static_cast<faiss::idx_t>(slot)) {
        invlists->release_ids(slot, ids);
        throw std::runtime_error("LIRE membership points to wrong list");
      }
    }
    invlists->release_ids(slot, ids);
    total += size;
  }
  if (total != static_cast<size_t>(ntotal) || active_count != active_nlist_) {
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
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  return reassign_moved_;
}

}  // namespace baseline_eval
