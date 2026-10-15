#include "faiss_ivfflat_lire.h"

#include <faiss/IndexFlat.h>
#include <faiss/invlists/InvertedLists.h>

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
      active_(config.max_nlist, 0) {
  if (d <= 0 || initial_nlist_ == 0 || max_nlist_ < initial_nlist_ ||
      config_.split_threshold < 2) {
    throw std::runtime_error("invalid IndexIVFFlatLIRE configuration");
  }
  own_fields = true;
  is_trained = false;
}

IndexIVFFlatLIRE::~IndexIVFFlatLIRE() = default;

IndexIVFFlatLIRE::ActiveCentroidIndex*
IndexIVFFlatLIRE::active_quantizer() {
  return static_cast<ActiveCentroidIndex*>(quantizer);
}

const IndexIVFFlatLIRE::ActiveCentroidIndex*
IndexIVFFlatLIRE::active_quantizer() const {
  return static_cast<const ActiveCentroidIndex*>(quantizer);
}

float IndexIVFFlatLIRE::l2(const float* a, const float* b) const {
  float distance = 0.0f;
  for (int j = 0; j < d; ++j) {
    const float diff = a[j] - b[j];
    distance += diff * diff;
  }
  return distance;
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
  std::mt19937 rng(config_.seed + static_cast<uint32_t>(split_serial_++));
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
    faiss::idx_t slot) {
  if (!active_[static_cast<size_t>(slot)] ||
      invlists->list_size(static_cast<size_t>(slot)) <=
          config_.split_threshold) {
    return {};
  }
  if (free_slots_.empty()) {
    throw std::runtime_error("LIRE max_nlist exhausted during split");
  }

  ListData input = read_list(slot);
  std::vector<uint8_t> labels;
  std::vector<float> new_centers;
  if (!split_balanced_kmeans(input, &labels, &new_centers)) return {};

  const std::vector<float> old_center(centroid(slot), centroid(slot) + d);
  const std::vector<faiss::idx_t> neighbors = nearest_centroid_slots(
      old_center.data(), slot, config_.reassign_k);
  const faiss::idx_t new_slot = free_slots_.back();
  free_slots_.pop_back();

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

  // Commit the split while the global write lock is held. Search cannot see a
  // centroid/list mismatch or a duplicate/missing vector during this phase.
  write_list(slot, children[0]);
  write_list(new_slot, children[1]);
  std::copy(new_centers.begin(), new_centers.begin() + d, centroid(slot));
  std::copy(new_centers.begin() + d, new_centers.end(), centroid(new_slot));
  active_[static_cast<size_t>(new_slot)] = 1;
  active_nlist_++;
  for (faiss::idx_t id : children[0].ids) membership_[static_cast<size_t>(id)] = slot;
  for (faiss::idx_t id : children[1].ids) membership_[static_cast<size_t>(id)] = new_slot;
  split_count_++;

  std::vector<faiss::idx_t> changed{slot, new_slot};
  if (config_.enable_reassign) {
    std::vector<faiss::idx_t> moved_targets = collect_and_reassign(
        slot, old_center, slot, new_slot, neighbors);
    changed.insert(changed.end(), moved_targets.begin(), moved_targets.end());
  }
  return changed;
}

std::vector<faiss::idx_t> IndexIVFFlatLIRE::collect_and_reassign(
    faiss::idx_t old_slot,
    const std::vector<float>& old_center,
    faiss::idx_t child0,
    faiss::idx_t child1,
    const std::vector<faiss::idx_t>& neighbors) {
  (void)old_slot;
  struct Candidate {
    faiss::idx_t id;
    faiss::idx_t source;
    std::vector<float> vector;
  };

  std::vector<Candidate> candidate_vectors;
  std::unordered_set<faiss::idx_t> candidate_ids;
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
    ListData list = read_list(child);
    for (size_t i = 0; i < list.size(); ++i) {
      const float* vector = list.row(i, d);
      reassign_scanned_++;
      // SPFresh CheckIsNeedReassign(..., isInSplitHead=true):
      // reject when D(v, old) >= D(v, current new head).
      if (l2(vector, old_center.data()) >= l2(vector, centroid(child))) continue;
      add_candidate(list.ids[i], child, vector);
    }
  };
  check_child(child0);
  check_child(child1);

  for (faiss::idx_t neighbor : neighbors) {
    if (!active_[static_cast<size_t>(neighbor)] || neighbor == child0 ||
        neighbor == child1) {
      continue;
    }
    ListData list = read_list(neighbor);
    for (size_t i = 0; i < list.size(); ++i) {
      const float* vector = list.row(i, d);
      reassign_scanned_++;
      const float old_distance = l2(vector, old_center.data());
      const float new0 = l2(vector, centroid(child0));
      const float new1 = l2(vector, centroid(child1));
      const float current = l2(vector, centroid(neighbor));
      // SPFresh CheckIsNeedReassign(..., isInSplitHead=false).
      if (old_distance <= new0 && old_distance <= new1) continue;
      if (current <= new0 && current <= new1) continue;
      add_candidate(list.ids[i], neighbor, vector);
    }
  }

  std::vector<Move> moves;
  moves.reserve(candidate_vectors.size());
  for (const Candidate& candidate : candidate_vectors) {
    if (membership_[static_cast<size_t>(candidate.id)] != candidate.source) {
      throw std::runtime_error("LIRE membership changed during reassign scan");
    }
    const faiss::idx_t target = nearest_active(candidate.vector.data());
    if (target == candidate.source) continue;
    Move move;
    move.id = candidate.id;
    move.source = candidate.source;
    move.target = target;
    move.vector = candidate.vector;
    moves.push_back(std::move(move));
  }

  apply_moves(moves);
  std::vector<faiss::idx_t> targets;
  targets.reserve(moves.size());
  for (const Move& move : moves) targets.push_back(move.target);
  std::sort(targets.begin(), targets.end());
  targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
  reassign_moved_ += moves.size();
  return targets;
}

void IndexIVFFlatLIRE::apply_moves(const std::vector<Move>& moves) {
  if (moves.empty()) return;
  std::unordered_map<faiss::idx_t, ListData> lists;
  std::unordered_map<faiss::idx_t, faiss::idx_t> target_by_id;
  for (const Move& move : moves) {
    if (!lists.count(move.source)) lists.emplace(move.source, read_list(move.source));
    if (!lists.count(move.target)) lists.emplace(move.target, read_list(move.target));
    target_by_id.emplace(move.id, move.target);
  }

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

  for (const Move& move : moves) {
    ListData& target = lists.at(move.target);
    target.ids.push_back(move.id);
    target.vectors.insert(target.vectors.end(), move.vector.begin(), move.vector.end());
  }
  for (const auto& entry : lists) write_list(entry.first, entry.second);
  for (const Move& move : moves) membership_[static_cast<size_t>(move.id)] = move.target;
}

void IndexIVFFlatLIRE::run_maintenance(
    const std::vector<faiss::idx_t>& touched) {
  std::deque<faiss::idx_t> queue;
  std::vector<uint8_t> queued(max_nlist_, 0);
  for (faiss::idx_t slot : touched) {
    if (slot < 0 || slot >= static_cast<faiss::idx_t>(max_nlist_)) continue;
    if (!queued[static_cast<size_t>(slot)] &&
        invlists->list_size(static_cast<size_t>(slot)) >
            config_.split_threshold) {
      queue.push_back(slot);
      queued[static_cast<size_t>(slot)] = 1;
    }
  }

  uint32_t splits = 0;
  while (!queue.empty()) {
    const faiss::idx_t slot = queue.front();
    queue.pop_front();
    queued[static_cast<size_t>(slot)] = 0;
    if (!active_[static_cast<size_t>(slot)] ||
        invlists->list_size(static_cast<size_t>(slot)) <=
            config_.split_threshold) {
      continue;
    }
    if (++splits > config_.max_splits_per_batch) {
      throw std::runtime_error("LIRE exceeded max_splits_per_batch");
    }
    const std::vector<faiss::idx_t> changed = split_one(slot);
    for (faiss::idx_t changed_slot : changed) {
      if (!queued[static_cast<size_t>(changed_slot)] &&
          invlists->list_size(static_cast<size_t>(changed_slot)) >
              config_.split_threshold) {
        queue.push_back(changed_slot);
        queued[static_cast<size_t>(changed_slot)] = 1;
      }
    }
  }
}

void IndexIVFFlatLIRE::train(faiss::idx_t n, const float* x) {
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
  rebuild_quantizer();
  is_trained = true;
}

void IndexIVFFlatLIRE::add_with_ids(
    faiss::idx_t n,
    const float* x,
    const faiss::idx_t* xids) {
  std::unique_lock<std::shared_mutex> lock(state_mutex_);
  if (!is_trained) throw std::runtime_error("LIRE index is not trained");
  if (n < 0 || (n > 0 && x == nullptr)) throw std::runtime_error("invalid LIRE add");
  if (n == 0) return;

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
  active_quantizer()->search(
      n, x, 1, coarse_distances.data(), assignments.data());

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

  membership_.resize(static_cast<size_t>(ntotal + n), -1);
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
      membership_[static_cast<size_t>(id)] = slot;
    }
    invlists->add_entries(
        static_cast<size_t>(slot),
        list_ids.size(),
        list_ids.data(),
        reinterpret_cast<const uint8_t*>(list_vectors.data()));
  }
  ntotal += n;

  run_maintenance(touched);
  rebuild_quantizer();
  if (config_.validate_invariants) validate_state();

  std::cout << "[LIRE] maintenance complete, active_clusters="
            << active_nlist_ << '\n';
}

void IndexIVFFlatLIRE::search(
    faiss::idx_t n,
    const float* x,
    faiss::idx_t k,
    float* distances,
    faiss::idx_t* labels,
    const faiss::SearchParameters* params) const {
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  faiss::IndexIVFFlat::search(n, x, k, distances, labels, params);
}

void IndexIVFFlatLIRE::reset() {
  std::unique_lock<std::shared_mutex> lock(state_mutex_);
  for (uint32_t slot = 0; slot < max_nlist_; ++slot) {
    invlists->resize(slot, 0);
  }
  membership_.clear();
  ntotal = 0;
  split_count_ = 0;
  reassign_scanned_ = 0;
  reassign_moved_ = 0;
}

size_t IndexIVFFlatLIRE::remove_ids(const faiss::IDSelector&) {
  throw std::runtime_error("IndexIVFFlatLIRE does not support deletion");
}

void IndexIVFFlatLIRE::validate_state() const {
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
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  return reassign_scanned_;
}

uint64_t IndexIVFFlatLIRE::reassign_moved() const {
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  return reassign_moved_;
}

}  // namespace baseline_eval