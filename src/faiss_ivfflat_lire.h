#pragma once

#include <faiss/IndexIVFFlat.h>

#include <cstdint>
#include <deque>
#include <shared_mutex>
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
  bool enable_reassign{true};
  bool validate_invariants{false};
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
  std::vector<faiss::idx_t> split_one(faiss::idx_t slot);
  std::vector<faiss::idx_t> collect_and_reassign(
      faiss::idx_t old_slot,
      const std::vector<float>& old_center,
      faiss::idx_t child0,
      faiss::idx_t child1,
      const std::vector<faiss::idx_t>& neighbors);
  void apply_moves(const std::vector<Move>& moves);
  void run_maintenance(const std::vector<faiss::idx_t>& touched);
  void validate_state() const;

  LireConfig config_;
  uint32_t initial_nlist_{0};
  uint32_t max_nlist_{0};
  uint32_t active_nlist_{0};
  std::vector<float> centroids_;
  std::vector<uint8_t> active_;
  std::vector<faiss::idx_t> free_slots_;
  std::vector<faiss::idx_t> membership_;
  uint64_t split_serial_{0};
  uint64_t split_count_{0};
  uint64_t reassign_scanned_{0};
  uint64_t reassign_moved_{0};
  mutable std::shared_mutex state_mutex_;
};

}  // namespace baseline_eval
