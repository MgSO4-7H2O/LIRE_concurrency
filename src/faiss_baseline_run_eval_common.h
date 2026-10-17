#pragma once

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>

#if defined(BASELINE_ENABLE_LIRE)
#include "faiss_ivfflat_lire.h"
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

namespace baseline_eval {

#if !defined(BASELINE_ALGO_NAME)
#error "BASELINE_ALGO_NAME must be defined before including this header"
#endif

#if !defined(BASELINE_INDEX_TYPE)
#error "BASELINE_INDEX_TYPE must be defined before including this header"
#endif

#if !defined(BASELINE_MODE_MS)
#define BASELINE_MODE_MS 0
#endif

constexpr size_t kGroundTruthBlockTargetBytes = static_cast<size_t>(64) << 20;

struct Config {
  std::string metric{"l2"};
  uint32_t dim{0};
  uint32_t topk{100};
  uint32_t nlist{8192};
  uint32_t nprobe{64};
  uint32_t max_queries{10000};
  uint32_t main_index_rows{200000};
  uint32_t snapshot_interval{10000};
  uint32_t final_rows{0};
  uint32_t train_rows{0};
  uint32_t seed{1337};
  uint32_t add_threads{0};
  uint32_t search_threads{0};
  uint32_t gt_threads{0};
  uint32_t M{16};
  uint32_t nbits{8};
  std::string sq_type{"QT_8bit"};
  uint32_t opq_M{0};
  uint32_t opq_niter{50};
  uint32_t opq_niter_pq{4};
  uint32_t opq_niter_pq_0{40};
  uint32_t opq_max_train_points{65536};
  uint32_t rerank_factor{BASELINE_INDEX_TYPE == 3 ? 1u : 10u};
  bool exact_rerank{true};
  bool unsafe_parallel_add{false};
  uint32_t nb_bits{1};
  uint32_t qb{4};
  bool centered{false};
#if defined(BASELINE_ENABLE_LIRE)
  std::string concurrent_gt_mode{"load"};
  std::string concurrent_dataset_path;
  std::string concurrent_gt_path;
  std::string concurrent_init_mode{"off"};
  std::string concurrent_init_path;
  std::string concurrent_summary_path;
  uint32_t concurrent_query_workers{8};
  uint32_t concurrent_query_threads_per_request{1};
  uint32_t concurrent_insert_threads{1};
  uint32_t concurrent_maintenance_threads{1};
  uint32_t concurrent_update_thread_budget{2};
  uint32_t concurrent_max_maintenance_backlog{0};
  uint64_t concurrent_max_insert_vectors{std::numeric_limits<uint64_t>::max()};
  double concurrent_insert_rate_limit{0.0};
  uint32_t lire_max_nlist{0};
  uint32_t lire_split_threshold{0};
  uint32_t lire_reassign_k{64};
  uint32_t lire_max_splits_per_batch{100000};
  bool lire_enable_reassign{true};
  bool lire_validate_invariants{false};
#endif
};

struct FvecsData {
  std::vector<float> values;
  uint32_t n{0};
  uint32_t d{0};
};

struct WorkloadOp {
  uint32_t op_id{0};
  std::string operation;
  uint32_t start{0};
  uint32_t end{0};
  uint32_t active_rows{0};
  int32_t round{-1};
  int32_t cluster{-1};
  uint32_t source_op_id{0};
};

struct Workload {
  std::string path;
  std::string dataset;
  uint32_t num_vectors{0};
  uint32_t dim{0};
  std::vector<WorkloadOp> operations;
};

struct SeriesStats {
  double avg{0.0};
  double p5{0.0};
  double p50{0.0};
  double p95{0.0};
  double p99{0.0};
  double min{0.0};
  double max{0.0};
};

struct RecallAgeMetrics {
  double recall_new{0.0};
  double recall_old{0.0};
  double gt_new_ratio{0.0};
  uint64_t gt_new_total{0};
  uint64_t gt_old_total{0};
  uint64_t hit_new_total{0};
  uint64_t hit_old_total{0};
};

struct EvalMetrics {
  double recall{0.0};
  double raw_recall{0.0};
  uint32_t rerank_candidates{0};
  RecallAgeMetrics age;
  double avg_query_ms{0.0};
  double p50_query_ms{0.0};
  double p99_query_ms{0.0};
  double qps{0.0};
  double faiss_search_ms{0.0};
  double rerank_ms{0.0};
  double query_eval_ms{0.0};
  double avg_scanned{0.0};
  uint32_t query_count{0};
};

struct Snapshot {
  uint32_t search_id{0};
  uint32_t op_id{0};
  uint32_t active_rows{0};
  uint32_t main_rows{0};
  uint32_t active_delta_rows{0};
  uint32_t frozen_delta_rows{0};
  int32_t round{-1};
  int32_t cluster{-1};
  uint32_t base_rows{0};
  uint32_t snapshot_rows{0};
  EvalMetrics metrics;
  double update_ms{0.0};
  double cumulative_update_ms{0.0};
  double snapshot_maintenance_ms{0.0};
  double snapshot_total_ms{0.0};
  double update_throughput_vecps{0.0};
  double amortized_update_throughput_vecps{0.0};
  bool skipped_warmup{false};
};

struct RunState {
  std::vector<faiss::idx_t> active_ids;
  uint32_t seen_rows{0};
  double total_update_ms{0.0};
};

std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c; break;
    }
  }
  return out;
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string ReadTextFile(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("Cannot open file: " + path);
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

std::string StripJsonComments(std::string text) {
  std::string out;
  out.reserve(text.size());
  bool in_string = false;
  bool escape = false;
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      out.push_back(c);
      if (escape) {
        escape = false;
      } else if (c == '\\') {
        escape = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      out.push_back(c);
      continue;
    }
    if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      while (i < text.size() && text[i] != '\n') ++i;
      if (i < text.size()) out.push_back('\n');
      continue;
    }
    out.push_back(c);
  }
  return out;
}

std::optional<std::string> JsonString(const std::string& text, const std::string& key) {
  const std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch m;
  if (std::regex_search(text, m, re)) return m[1].str();
  return std::nullopt;
}

std::optional<double> JsonNumber(const std::string& text, const std::string& key) {
  const std::regex re("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
  std::smatch m;
  if (std::regex_search(text, m, re)) return std::stod(m[1].str());
  return std::nullopt;
}

std::optional<bool> JsonBool(const std::string& text, const std::string& key) {
  const std::regex re("\\\"" + key + "\\\"\\s*:\\s*(true|false|0|1)");
  std::smatch m;
  if (!std::regex_search(text, m, re)) return std::nullopt;
  const std::string v = m[1].str();
  return v == "true" || v == "1";
}

uint32_t JsonUint(const std::string& text, const std::string& key, uint32_t def) {
  if (auto v = JsonNumber(text, key)) return static_cast<uint32_t>(*v);
  return def;
}

bool JsonBoolDefault(const std::string& text, const std::string& key, bool def) {
  if (auto v = JsonBool(text, key)) return *v;
  return def;
}

Config LoadConfigJson(const std::string& path) {
  const std::string text = StripJsonComments(ReadTextFile(path));
  Config cfg;
  if (auto v = JsonString(text, "metric")) cfg.metric = *v;
  if (auto v = JsonString(text, "distance")) cfg.metric = *v;
  if (auto v = JsonString(text, "distance_metric")) cfg.metric = *v;
  cfg.topk = JsonUint(text, "topk", JsonUint(text, "k", cfg.topk));
  cfg.nlist = JsonUint(text, "nlist", cfg.nlist);
  cfg.nprobe = JsonUint(text, "nprobe", cfg.nprobe);
  cfg.max_queries = JsonUint(text, "max_queries", cfg.max_queries);
  cfg.main_index_rows = JsonUint(text, "main_index_rows", cfg.main_index_rows);
  cfg.snapshot_interval = JsonUint(text, "snapshot_interval", cfg.snapshot_interval);
  cfg.final_rows = JsonUint(text, "final_rows", cfg.final_rows);
  cfg.train_rows = JsonUint(text, "train_rows", cfg.train_rows);
  cfg.seed = JsonUint(text, "seed", cfg.seed);
  cfg.add_threads = JsonUint(text, "add_threads", cfg.add_threads);
  cfg.search_threads = JsonUint(text, "search_threads", cfg.search_threads);
  cfg.gt_threads = JsonUint(text, "gt_threads", cfg.gt_threads);
  cfg.M = JsonUint(text, "M", JsonUint(text, "m", cfg.M));
  cfg.nbits = JsonUint(text, "nbits", cfg.nbits);
  if (auto v = JsonString(text, "sq_type")) cfg.sq_type = *v;
  cfg.opq_M = JsonUint(text, "opq_M", cfg.opq_M);
  cfg.opq_niter = JsonUint(text, "opq_niter", cfg.opq_niter);
  cfg.opq_niter_pq = JsonUint(text, "opq_niter_pq", cfg.opq_niter_pq);
  cfg.opq_niter_pq_0 = JsonUint(text, "opq_niter_pq_0", cfg.opq_niter_pq_0);
  cfg.opq_max_train_points = JsonUint(text, "opq_max_train_points", cfg.opq_max_train_points);
  cfg.rerank_factor = JsonUint(text, "rerank_factor", cfg.rerank_factor);
  cfg.exact_rerank = JsonBoolDefault(text, "exact_rerank", cfg.exact_rerank);
  cfg.unsafe_parallel_add = JsonBoolDefault(text, "unsafe_parallel_add", cfg.unsafe_parallel_add);
  cfg.nb_bits = JsonUint(text, "nb_bits", cfg.nb_bits);
  cfg.qb = JsonUint(text, "qb", cfg.qb);
  cfg.centered = JsonBoolDefault(text, "centered", cfg.centered);
#if defined(BASELINE_ENABLE_LIRE)
  if (auto v = JsonString(text, "concurrent_gt_mode")) cfg.concurrent_gt_mode = *v;
  if (auto v = JsonString(text, "concurrent_dataset_path")) cfg.concurrent_dataset_path = *v;
  if (auto v = JsonString(text, "concurrent_gt_path")) cfg.concurrent_gt_path = *v;
  if (auto v = JsonString(text, "concurrent_init_mode")) cfg.concurrent_init_mode = *v;
  if (auto v = JsonString(text, "concurrent_init_path")) cfg.concurrent_init_path = *v;
  if (auto v = JsonString(text, "concurrent_summary_path")) cfg.concurrent_summary_path = *v;
  cfg.concurrent_query_workers = JsonUint(text, "concurrent_query_workers", cfg.concurrent_query_workers);
  cfg.concurrent_query_threads_per_request = JsonUint(text, "concurrent_query_threads_per_request", cfg.concurrent_query_threads_per_request);
  cfg.concurrent_insert_threads = JsonUint(text, "concurrent_insert_threads", cfg.concurrent_insert_threads);
  cfg.concurrent_maintenance_threads = JsonUint(text, "concurrent_maintenance_threads", cfg.concurrent_maintenance_threads);
  cfg.concurrent_update_thread_budget = JsonUint(text, "concurrent_update_thread_budget", cfg.concurrent_update_thread_budget);
  cfg.concurrent_max_maintenance_backlog = JsonUint(text, "concurrent_max_maintenance_backlog", cfg.concurrent_max_maintenance_backlog);
  if (auto v = JsonNumber(text, "concurrent_max_insert_vectors")) cfg.concurrent_max_insert_vectors = static_cast<uint64_t>(*v);
  if (auto v = JsonNumber(text, "concurrent_insert_rate_limit_vecps")) cfg.concurrent_insert_rate_limit = *v;
  cfg.lire_max_nlist = JsonUint(text, "lire_max_nlist", cfg.lire_max_nlist);
  cfg.lire_split_threshold = JsonUint(
      text, "lire_split_threshold", cfg.lire_split_threshold);
  cfg.lire_reassign_k = JsonUint(text, "lire_reassign_k", cfg.lire_reassign_k);
  cfg.lire_max_splits_per_batch = JsonUint(
      text, "lire_max_splits_per_batch", cfg.lire_max_splits_per_batch);
  cfg.lire_enable_reassign = JsonBoolDefault(
      text, "lire_enable_reassign", cfg.lire_enable_reassign);
  cfg.lire_validate_invariants = JsonBoolDefault(
      text, "lire_validate_invariants", cfg.lire_validate_invariants);
#endif

#ifdef _OPENMP
  const uint32_t max_threads = static_cast<uint32_t>(std::max(1, omp_get_max_threads()));
#else
  const uint32_t max_threads = 1;
#endif
  if (cfg.add_threads == 0) cfg.add_threads = max_threads;
  if (cfg.search_threads == 0) cfg.search_threads = max_threads;
  if (cfg.gt_threads == 0) cfg.gt_threads = max_threads;
  return cfg;
}

std::string ResolveFileWithSuffix(const std::string& path_or_dir,
                                  const std::string& required_suffix) {
  fs::path p(path_or_dir);
  if (!fs::exists(p)) throw std::runtime_error("Path does not exist: " + path_or_dir);
  if (fs::is_regular_file(p)) {
    if (!required_suffix.empty() && !EndsWith(p.filename().string(), required_suffix)) {
      throw std::runtime_error("Expected file ending with " + required_suffix + ": " + p.string());
    }
    return p.string();
  }
  if (fs::is_directory(p)) {
    std::vector<fs::path> matches;
    for (const auto& entry : fs::directory_iterator(p)) {
      if (!entry.is_regular_file()) continue;
      const std::string name = entry.path().filename().string();
      if (required_suffix.empty() || EndsWith(name, required_suffix)) matches.push_back(entry.path());
    }
    if (matches.empty()) {
      throw std::runtime_error("No file ending with " + required_suffix + " found under " + path_or_dir);
    }
    std::sort(matches.begin(), matches.end());
    return matches.front().string();
  }
  throw std::runtime_error("Unsupported path type: " + path_or_dir);
}

std::filesystem::path BuildDefaultResultPath(const std::string& config_path,
                                             const std::string& base_path) {
  const std::string config_name = fs::path(config_path).stem().string();
  std::string dataset_name = fs::path(base_path).parent_path().filename().string();
  if (dataset_name.empty()) dataset_name = fs::path(base_path).stem().string();
  std::string metric_name = "default";
  fs::path p(base_path);
  std::vector<std::string> parts;
  for (const auto& part : p) parts.push_back(part.string());
  for (size_t i = 0; i < parts.size(); ++i) {
    if (parts[i] == "data" && i + 2 < parts.size()) {
      dataset_name = parts[i + 1];
      std::string metric;
      for (size_t j = i + 2; j + 1 < parts.size(); ++j) {
        if (!metric.empty()) metric += "_";
        metric += parts[j];
      }
      if (!metric.empty()) metric_name = metric;
      break;
    }
  }
  fs::path dir = fs::path("result") / dataset_name / metric_name /
                 BASELINE_ALGO_NAME / config_name;
  fs::create_directories(dir);
  return dir / "online_eval.json";
}

FvecsData LoadFvecs(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) throw std::runtime_error("Failed to open fvecs file: " + path);
  FvecsData out;
  int32_t dim = -1;
  while (true) {
    int32_t cur_dim = 0;
    ifs.read(reinterpret_cast<char*>(&cur_dim), sizeof(int32_t));
    if (!ifs) {
      if (ifs.eof() && ifs.gcount() == 0) break;
      throw std::runtime_error("Failed to read dimension from: " + path);
    }
    if (cur_dim <= 0) throw std::runtime_error("Invalid dimension in fvecs file: " + path);
    if (dim == -1) dim = cur_dim;
    if (cur_dim != dim) throw std::runtime_error("Mixed dimensions in fvecs file: " + path);
    const size_t old_size = out.values.size();
    out.values.resize(old_size + static_cast<size_t>(cur_dim));
    ifs.read(reinterpret_cast<char*>(out.values.data() + old_size),
             sizeof(float) * static_cast<size_t>(cur_dim));
    if (!ifs) throw std::runtime_error("Failed to read payload from: " + path);
    out.n += 1;
  }
  if (dim <= 0 || out.n == 0) throw std::runtime_error("fvecs file contains no vectors: " + path);
  out.d = static_cast<uint32_t>(dim);
  return out;
}

inline const float* RowPtr(const FvecsData& x, uint32_t i) {
  return x.values.data() + static_cast<size_t>(i) * x.d;
}

void LimitQueries(FvecsData* query, uint32_t max_queries) {
  if (!query || max_queries == 0 || query->n <= max_queries) return;
  query->values.resize(static_cast<size_t>(max_queries) * query->d);
  query->n = max_queries;
}

bool IsCosineMetric(const std::string& metric) {
  std::string m = metric;
  std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return std::tolower(c); });
  return m == "cosine" || m == "cos" || m == "angular";
}

bool IsL2Metric(const std::string& metric) {
  std::string m = metric;
  std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return std::tolower(c); });
  return m == "l2" || m == "euclidean" || m == "sq_l2" || m == "squared_l2";
}

void NormalizeRowsL2(FvecsData* x) {
  if (!x) return;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(x->n); ++i) {
    float* row = x->values.data() + static_cast<size_t>(i) * x->d;
    double ss = 0.0;
    for (uint32_t j = 0; j < x->d; ++j) ss += static_cast<double>(row[j]) * row[j];
    const float norm = static_cast<float>(std::sqrt(ss));
    if (norm > 0.0f) {
      for (uint32_t j = 0; j < x->d; ++j) row[j] /= norm;
    }
  }
}

double Percentile(std::vector<double> values, double q) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double idx = q * static_cast<double>(values.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(idx));
  size_t hi = static_cast<size_t>(std::ceil(idx));
  if (hi >= values.size()) hi = values.size() - 1;
  const double frac = idx - static_cast<double>(lo);
  return values[lo] + (values[hi] - values[lo]) * frac;
}

SeriesStats SummarizeSeries(const std::vector<double>& values) {
  SeriesStats s;
  if (values.empty()) return s;
  s.avg = std::accumulate(values.begin(), values.end(), 0.0) /
          static_cast<double>(values.size());
  s.p5 = Percentile(values, 0.05);
  s.p50 = Percentile(values, 0.50);
  s.p95 = Percentile(values, 0.95);
  s.p99 = Percentile(values, 0.99);
  s.min = *std::min_element(values.begin(), values.end());
  s.max = *std::max_element(values.begin(), values.end());
  return s;
}

uint64_t ReadProcStatusBytes(const std::string& key) {
  std::ifstream ifs("/proc/self/status");
  if (!ifs) return 0;
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.rfind(key, 0) != 0) continue;
    std::istringstream iss(line.substr(key.size()));
    uint64_t kb = 0;
    std::string unit;
    iss >> kb >> unit;
    return kb * 1024ull;
  }
  return 0;
}

faiss::ScalarQuantizer::QuantizerType ParseSqType(const std::string& s) {
  if (s == "QT_8bit") return faiss::ScalarQuantizer::QT_8bit;
  if (s == "QT_4bit") return faiss::ScalarQuantizer::QT_4bit;
  if (s == "QT_6bit") return faiss::ScalarQuantizer::QT_6bit;
  if (s == "QT_fp16") return faiss::ScalarQuantizer::QT_fp16;
  if (s == "QT_8bit_uniform") return faiss::ScalarQuantizer::QT_8bit_uniform;
  throw std::runtime_error(
      "Unsupported sq_type: " + s +
      " (supported: QT_8bit, QT_4bit, QT_6bit, QT_fp16, QT_8bit_uniform)");
}

std::optional<std::string> ExtractStringField(const std::string& obj, const std::string& key) {
  const std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch m;
  if (std::regex_search(obj, m, re)) return m[1].str();
  return std::nullopt;
}

std::optional<uint32_t> ExtractUintField(const std::string& obj, const std::string& key) {
  const std::regex re("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
  std::smatch m;
  if (std::regex_search(obj, m, re)) return static_cast<uint32_t>(std::stoul(m[1].str()));
  return std::nullopt;
}

std::optional<int32_t> ExtractIntOrNullField(const std::string& obj, const std::string& key) {
  const std::regex re("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+|null)");
  std::smatch m;
  if (!std::regex_search(obj, m, re)) return std::nullopt;
  if (m[1].str() == "null") return std::nullopt;
  return static_cast<int32_t>(std::stol(m[1].str()));
}

Workload LoadWorkloadJson(const std::string& path) {
  Workload wl;
  wl.path = path;
  const std::string text = ReadTextFile(path);
  if (auto v = ExtractStringField(text, "dataset")) wl.dataset = *v;
  if (auto v = ExtractUintField(text, "num_vectors")) wl.num_vectors = *v;
  if (auto v = ExtractUintField(text, "dim")) wl.dim = *v;

  const size_t ops_pos = text.find("\"operations\"");
  if (ops_pos == std::string::npos) throw std::runtime_error("workload JSON does not contain operations");
  const size_t arr_begin = text.find('[', ops_pos);
  if (arr_begin == std::string::npos) throw std::runtime_error("workload JSON operations is not an array");
  int depth = 0;
  size_t arr_end = std::string::npos;
  for (size_t i = arr_begin; i < text.size(); ++i) {
    if (text[i] == '[') depth++;
    if (text[i] == ']') {
      depth--;
      if (depth == 0) { arr_end = i; break; }
    }
  }
  if (arr_end == std::string::npos) throw std::runtime_error("unterminated operations array");
  const std::string arr = text.substr(arr_begin + 1, arr_end - arr_begin - 1);
  size_t pos = 0;
  while (true) {
    const size_t begin = arr.find('{', pos);
    if (begin == std::string::npos) break;
    int obj_depth = 0;
    size_t end = std::string::npos;
    for (size_t i = begin; i < arr.size(); ++i) {
      if (arr[i] == '{') obj_depth++;
      if (arr[i] == '}') {
        obj_depth--;
        if (obj_depth == 0) { end = i; break; }
      }
    }
    if (end == std::string::npos) throw std::runtime_error("unterminated operation object");
    const std::string obj = arr.substr(begin, end - begin + 1);
    WorkloadOp op;
    auto op_name = ExtractStringField(obj, "operation");
    if (!op_name) throw std::runtime_error("workload op missing operation");
    op.operation = *op_name;
    op.op_id = ExtractUintField(obj, "op_id").value_or(static_cast<uint32_t>(wl.operations.size()));
    op.start = ExtractUintField(obj, "start").value_or(0);
    op.end = ExtractUintField(obj, "end").value_or(op.start);
    op.active_rows = ExtractUintField(obj, "active_rows").value_or(op.end);
    op.round = ExtractIntOrNullField(obj, "round").value_or(-1);
    op.cluster = ExtractIntOrNullField(obj, "cluster").value_or(-1);
    op.source_op_id = ExtractUintField(obj, "source_op_id").value_or(0);
    wl.operations.push_back(op);
    pos = end + 1;
  }
  if (wl.operations.empty()) throw std::runtime_error("no operations parsed from workload JSON");
  return wl;
}

struct ExactCandidate {
  float dist{0.0f};
  faiss::idx_t doc_id{-1};
};

bool BetterExact(const ExactCandidate& a, const ExactCandidate& b) {
  if (a.dist != b.dist) return a.dist < b.dist;
  return a.doc_id < b.doc_id;
}

std::vector<std::vector<faiss::idx_t>> ExactTopKBlockwise(const FvecsData& base,
                                                          uint32_t active_rows,
                                                          const FvecsData& query,
                                                          uint32_t topk,
                                                          uint32_t num_threads) {
  if (active_rows == 0) throw std::runtime_error("ExactTopKBlockwise: no active rows");
  const uint32_t use_topk = std::min<uint32_t>(topk, active_rows);
  std::vector<std::vector<faiss::idx_t>> gt(query.n, std::vector<faiss::idx_t>(use_topk, -1));
#ifdef _OPENMP
#pragma omp parallel for num_threads((int)num_threads) schedule(dynamic)
#endif
  for (int64_t qi64 = 0; qi64 < static_cast<int64_t>(query.n); ++qi64) {
    const uint32_t qi = static_cast<uint32_t>(qi64);
    const float* q = RowPtr(query, qi);
    auto cmp = [](const ExactCandidate& a, const ExactCandidate& b) { return BetterExact(a, b); };
    std::priority_queue<ExactCandidate, std::vector<ExactCandidate>, decltype(cmp)> heap(cmp);
    for (uint32_t doc = 0; doc < active_rows; ++doc) {
      const float* x = RowPtr(base, doc);
      float dist = 0.0f;
      for (uint32_t j = 0; j < base.d; ++j) {
        const float diff = q[j] - x[j];
        dist += diff * diff;
      }
      ExactCandidate cand{dist, static_cast<faiss::idx_t>(doc)};
      if (heap.size() < use_topk) heap.push(cand);
      else if (BetterExact(cand, heap.top())) { heap.pop(); heap.push(cand); }
    }
    std::vector<ExactCandidate> tmp;
    tmp.reserve(heap.size());
    while (!heap.empty()) { tmp.push_back(heap.top()); heap.pop(); }
    std::sort(tmp.begin(), tmp.end(), BetterExact);
    for (uint32_t k = 0; k < use_topk; ++k) gt[qi][k] = tmp[k].doc_id;
  }
  return gt;
}

double RecallAtK(const std::vector<std::vector<faiss::idx_t>>& gt,
                 const std::vector<faiss::idx_t>& pred,
                 uint32_t nq,
                 uint32_t pred_stride,
                 uint32_t topk) {
  uint64_t total_hits = 0;
  uint64_t total = 0;
  for (uint32_t qi = 0; qi < nq; ++qi) {
    const auto& gt_row = gt[qi];
    const uint32_t denom_k = std::min<uint32_t>(topk, static_cast<uint32_t>(gt_row.size()));
    total += denom_k;
    for (uint32_t k = 0; k < denom_k; ++k) {
      const faiss::idx_t doc_id = pred[static_cast<size_t>(qi) * pred_stride + k];
      if (doc_id < 0) continue;
      if (std::find(gt_row.begin(), gt_row.end(), doc_id) != gt_row.end()) total_hits++;
    }
  }
  return total > 0 ? static_cast<double>(total_hits) / static_cast<double>(total) : 0.0;
}

RecallAgeMetrics ComputeRecallByRecentInsert(const std::vector<std::vector<faiss::idx_t>>& gt,
                                             const std::vector<faiss::idx_t>& pred,
                                             uint32_t nq,
                                             uint32_t pred_stride,
                                             uint32_t topk,
                                             uint32_t new_begin,
                                             uint32_t new_end,
                                             uint32_t seen_rows) {
  RecallAgeMetrics out;
  if (new_begin > new_end || new_end > seen_rows) {
    new_begin = seen_rows;
    new_end = seen_rows;
  }
  for (uint32_t qi = 0; qi < nq; ++qi) {
    std::unordered_set<faiss::idx_t> pred_set;
    pred_set.reserve(static_cast<size_t>(topk) * 2 + 1);
    for (uint32_t k = 0; k < topk; ++k) {
      const faiss::idx_t doc = pred[static_cast<size_t>(qi) * pred_stride + k];
      if (doc >= 0) pred_set.insert(doc);
    }
    for (faiss::idx_t doc : gt[qi]) {
      if (doc < 0 || doc >= static_cast<faiss::idx_t>(seen_rows)) continue;
      const bool is_new = doc >= static_cast<faiss::idx_t>(new_begin) &&
                          doc < static_cast<faiss::idx_t>(new_end);
      const bool hit = pred_set.find(doc) != pred_set.end();
      if (is_new) {
        out.gt_new_total++;
        if (hit) out.hit_new_total++;
      } else {
        out.gt_old_total++;
        if (hit) out.hit_old_total++;
      }
    }
  }
  const uint64_t gt_total = out.gt_new_total + out.gt_old_total;
  out.recall_new = out.gt_new_total > 0 ? static_cast<double>(out.hit_new_total) / out.gt_new_total : 0.0;
  out.recall_old = out.gt_old_total > 0 ? static_cast<double>(out.hit_old_total) / out.gt_old_total : 0.0;
  out.gt_new_ratio = gt_total > 0 ? static_cast<double>(out.gt_new_total) / gt_total : 0.0;
  return out;
}

std::vector<faiss::idx_t> ExactRerankCandidates(const FvecsData& base,
                                                const FvecsData& query,
                                                const std::vector<faiss::idx_t>& candidate_labels,
                                                uint32_t candidate_k,
                                                uint32_t final_k,
                                                uint32_t num_threads) {
  std::vector<faiss::idx_t> out(static_cast<size_t>(query.n) * final_k, -1);
#ifdef _OPENMP
#pragma omp parallel for num_threads((int)num_threads) schedule(dynamic)
#endif
  for (int64_t qi64 = 0; qi64 < static_cast<int64_t>(query.n); ++qi64) {
    const uint32_t qi = static_cast<uint32_t>(qi64);
    const float* q = RowPtr(query, qi);
    std::vector<ExactCandidate> scored;
    scored.reserve(candidate_k);
    std::unordered_set<faiss::idx_t> seen;
    seen.reserve(static_cast<size_t>(candidate_k) * 2 + 1);
    for (uint32_t ck = 0; ck < candidate_k; ++ck) {
      const faiss::idx_t doc_id = candidate_labels[static_cast<size_t>(qi) * candidate_k + ck];
      if (doc_id < 0 || doc_id >= static_cast<faiss::idx_t>(base.n)) continue;
      if (!seen.insert(doc_id).second) continue;
      const float* x = RowPtr(base, static_cast<uint32_t>(doc_id));
      float dist = 0.0f;
      for (uint32_t j = 0; j < base.d; ++j) {
        const float diff = q[j] - x[j];
        dist += diff * diff;
      }
      scored.push_back(ExactCandidate{dist, doc_id});
    }
    std::sort(scored.begin(), scored.end(), BetterExact);
    const uint32_t keep = std::min<uint32_t>(final_k, static_cast<uint32_t>(scored.size()));
    for (uint32_t k = 0; k < keep; ++k) out[static_cast<size_t>(qi) * final_k + k] = scored[k].doc_id;
  }
  return out;
}
std::unique_ptr<faiss::Index> CreateConfiguredIndex(const Config& cfg,
                                                    faiss::IndexFlatL2* quantizer) {
#if BASELINE_INDEX_TYPE == 0
#if defined(BASELINE_ENABLE_LIRE)
  LireConfig lire;
  lire.initial_nlist = cfg.nlist;
  lire.max_nlist = cfg.lire_max_nlist;
  lire.split_threshold = cfg.lire_split_threshold;
  lire.reassign_k = cfg.lire_reassign_k;
  lire.seed = cfg.seed;
  lire.max_splits_per_batch = cfg.lire_max_splits_per_batch;
  lire.enable_reassign = cfg.lire_enable_reassign;
  lire.validate_invariants = cfg.lire_validate_invariants;
  (void)quantizer;
  return std::make_unique<IndexIVFFlatLIRE>(static_cast<int>(cfg.dim), lire);
#else
  return std::make_unique<faiss::IndexIVFFlat>(quantizer,
                                               static_cast<int>(cfg.dim),
                                               static_cast<faiss::idx_t>(cfg.nlist),
                                               faiss::METRIC_L2);
#endif
#elif BASELINE_INDEX_TYPE == 1
  return std::make_unique<faiss::IndexIVFPQ>(quantizer,
                                             static_cast<size_t>(cfg.dim),
                                             static_cast<size_t>(cfg.nlist),
                                             static_cast<size_t>(cfg.M),
                                             static_cast<size_t>(cfg.nbits),
                                             faiss::METRIC_L2);
#elif BASELINE_INDEX_TYPE == 2
  return std::make_unique<faiss::IndexIVFRaBitQ>(quantizer,
                                                 static_cast<size_t>(cfg.dim),
                                                 static_cast<size_t>(cfg.nlist),
                                                 faiss::METRIC_L2,
                                                 true,
                                                 static_cast<uint8_t>(cfg.nb_bits));
#elif BASELINE_INDEX_TYPE == 3
  return std::make_unique<faiss::IndexIVFScalarQuantizer>(
      quantizer,
      static_cast<int>(cfg.dim),
      static_cast<faiss::idx_t>(cfg.nlist),
      ParseSqType(cfg.sq_type),
      faiss::METRIC_L2,
      true);
#elif BASELINE_INDEX_TYPE == 4
  auto* opq_quantizer = new faiss::IndexFlatL2(static_cast<int>(cfg.dim));
  auto* ivfpq = new faiss::IndexIVFPQ(opq_quantizer,
                                      static_cast<size_t>(cfg.dim),
                                      static_cast<size_t>(cfg.nlist),
                                      static_cast<size_t>(cfg.M),
                                      static_cast<size_t>(cfg.nbits),
                                      faiss::METRIC_L2);
  ivfpq->own_fields = true;
  auto* opq = new faiss::OPQMatrix(static_cast<int>(cfg.dim),
                                   static_cast<int>(cfg.opq_M == 0 ? cfg.M : cfg.opq_M));
  opq->niter = static_cast<int>(cfg.opq_niter);
  opq->niter_pq = static_cast<int>(cfg.opq_niter_pq);
  opq->niter_pq_0 = static_cast<int>(cfg.opq_niter_pq_0);
  opq->max_train_points = static_cast<size_t>(cfg.opq_max_train_points);
  auto* pre = new faiss::IndexPreTransform(opq, ivfpq);
  pre->own_fields = true;
  return std::unique_ptr<faiss::Index>(pre);
#endif
}

uint32_t CandidateK(const Config& cfg, uint32_t active_rows) {
#if BASELINE_INDEX_TYPE == 0
  (void)cfg;
  return std::min<uint32_t>(active_rows, cfg.topk);
#else
  const uint32_t factor = cfg.exact_rerank ? std::max<uint32_t>(1, cfg.rerank_factor) : 1;
  const uint64_t requested = static_cast<uint64_t>(cfg.topk) * factor;
  return static_cast<uint32_t>(std::min<uint64_t>(active_rows, requested));
#endif
}

uint32_t AddThreadsForIndex(const Config& cfg) {
#if BASELINE_INDEX_TYPE == 1 || BASELINE_INDEX_TYPE == 4
  return cfg.unsafe_parallel_add ? cfg.add_threads : 1;
#else
  return cfg.add_threads;
#endif
}

void SetIndexParams(faiss::Index& index, const Config& cfg) {
  auto* ivf = dynamic_cast<faiss::IndexIVF*>(&index);
  if (ivf) ivf->nprobe = static_cast<size_t>(cfg.nprobe);
  auto* pre = dynamic_cast<faiss::IndexPreTransform*>(&index);
  if (pre) {
    auto* pre_ivf = dynamic_cast<faiss::IndexIVF*>(pre->index);
    if (pre_ivf) pre_ivf->nprobe = static_cast<size_t>(cfg.nprobe);
  }
#if BASELINE_INDEX_TYPE == 2
  auto* rabitq = dynamic_cast<faiss::IndexIVFRaBitQ*>(&index);
  if (rabitq) {
    rabitq->nprobe = static_cast<size_t>(cfg.nprobe);
    rabitq->qb = static_cast<uint8_t>(std::min<uint32_t>(cfg.qb, 255));
  }
#endif
}

void SearchIndex(faiss::Index& index,
                 const Config& cfg,
                 uint32_t nq,
                 uint32_t candidate_k,
                 const float* queries,
                 float* distances,
                 faiss::idx_t* labels) {
#if BASELINE_INDEX_TYPE == 2
  auto* rabitq = dynamic_cast<faiss::IndexIVFRaBitQ*>(&index);
  if (!rabitq) throw std::runtime_error("Expected IndexIVFRaBitQ");
  faiss::IVFRaBitQSearchParameters sp;
  sp.nprobe = static_cast<size_t>(cfg.nprobe);
  sp.qb = static_cast<uint8_t>(std::min<uint32_t>(cfg.qb, 255));
  sp.centered = cfg.centered;
  rabitq->search(static_cast<faiss::idx_t>(nq), queries, candidate_k, distances, labels, &sp);
#else
  (void)cfg;
  index.search(static_cast<faiss::idx_t>(nq), queries, candidate_k, distances, labels);
#endif
}

EvalMetrics EvaluateSnapshot(faiss::Index& index,
                             const FvecsData& base,
                             const FvecsData& query,
                             uint32_t active_rows,
                             uint32_t last_insert_begin,
                             uint32_t last_insert_end,
                             const Config& cfg) {
  EvalMetrics m;
  m.query_count = query.n;
  if (active_rows == 0) return m;
  const uint32_t eval_k = std::min<uint32_t>(cfg.topk, active_rows);
  const uint32_t candidate_k = std::max<uint32_t>(eval_k, CandidateK(cfg, active_rows));
  m.rerank_candidates = candidate_k;

  const auto gt = ExactTopKBlockwise(base, active_rows, query, cfg.topk, cfg.gt_threads);

  SetIndexParams(index, cfg);
  std::vector<float> distances(static_cast<size_t>(query.n) * candidate_k, 0.0f);
  std::vector<faiss::idx_t> labels(static_cast<size_t>(query.n) * candidate_k, -1);

#ifdef _OPENMP
  const int old_threads = omp_get_max_threads();
  omp_set_num_threads(static_cast<int>(cfg.search_threads));
#endif
  faiss::indexIVF_stats.reset();
  const size_t ndis_before = faiss::indexIVF_stats.ndis;
  auto s0 = std::chrono::high_resolution_clock::now();
  SearchIndex(index, cfg, query.n, candidate_k, query.values.data(), distances.data(), labels.data());
  auto s1 = std::chrono::high_resolution_clock::now();
  const size_t ndis_after = faiss::indexIVF_stats.ndis;
#ifdef _OPENMP
  omp_set_num_threads(old_threads);
#endif

  m.faiss_search_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
  m.raw_recall = RecallAtK(gt, labels, query.n, candidate_k, eval_k);

  std::vector<faiss::idx_t> final_labels;
  auto r0 = std::chrono::high_resolution_clock::now();
#if BASELINE_INDEX_TYPE == 0
  final_labels.resize(static_cast<size_t>(query.n) * eval_k, -1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t qi = 0; qi < static_cast<int64_t>(query.n); ++qi) {
    for (uint32_t k = 0; k < eval_k; ++k) {
      final_labels[static_cast<size_t>(qi) * eval_k + k] = labels[static_cast<size_t>(qi) * candidate_k + k];
    }
  }
#else
  if (cfg.exact_rerank && candidate_k > eval_k) {
    final_labels = ExactRerankCandidates(base, query, labels, candidate_k, eval_k, cfg.gt_threads);
  } else {
    final_labels.resize(static_cast<size_t>(query.n) * eval_k, -1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t qi = 0; qi < static_cast<int64_t>(query.n); ++qi) {
      for (uint32_t k = 0; k < eval_k; ++k) {
        final_labels[static_cast<size_t>(qi) * eval_k + k] = labels[static_cast<size_t>(qi) * candidate_k + k];
      }
    }
  }
#endif
  auto r1 = std::chrono::high_resolution_clock::now();
  m.rerank_ms = std::chrono::duration<double, std::milli>(r1 - r0).count();
#if BASELINE_INDEX_TYPE == 0
  m.rerank_ms = 0.0;
#endif
  m.recall = RecallAtK(gt, final_labels, query.n, eval_k, eval_k);
  m.age = ComputeRecallByRecentInsert(gt, final_labels, query.n, eval_k, eval_k,
                                      last_insert_begin, last_insert_end, active_rows);
  m.query_eval_ms = m.faiss_search_ms + m.rerank_ms;
  m.avg_query_ms = query.n > 0 ? m.query_eval_ms / static_cast<double>(query.n) : 0.0;
  m.p50_query_ms = m.avg_query_ms;
  m.p99_query_ms = m.avg_query_ms;
  m.qps = m.query_eval_ms > 0.0 ? static_cast<double>(query.n) / (m.query_eval_ms / 1000.0) : 0.0;
  m.avg_scanned = query.n > 0 ? static_cast<double>(ndis_after - ndis_before) / static_cast<double>(query.n) : 0.0;
  return m;
}

void ValidateConfig(const Config& cfg, const FvecsData& base) {
  if (!IsL2Metric(cfg.metric) && !IsCosineMetric(cfg.metric)) {
    throw std::runtime_error("metric must be l2 or cosine, got: " + cfg.metric);
  }
  if (cfg.topk == 0) throw std::runtime_error("topk must be > 0");
  if (cfg.nlist == 0) throw std::runtime_error("nlist must be > 0");
  if (cfg.nprobe == 0 || cfg.nprobe > cfg.nlist) throw std::runtime_error("nprobe must be in [1, nlist]");
  if (!BASELINE_MODE_MS && cfg.main_index_rows > base.n) throw std::runtime_error("main_index_rows exceeds base rows");
  if (!BASELINE_MODE_MS && cfg.main_index_rows == 0) throw std::runtime_error("main_index_rows must be > 0");
  if (!BASELINE_MODE_MS && cfg.snapshot_interval == 0) throw std::runtime_error("snapshot_interval must be > 0");
#if BASELINE_INDEX_TYPE == 1
  if (cfg.M == 0) throw std::runtime_error("M must be > 0 for IVFPQ");
  if (cfg.nbits == 0 || cfg.nbits > 16) throw std::runtime_error("nbits must be in [1,16] for IVFPQ");
  if (cfg.dim % cfg.M != 0) throw std::runtime_error("dim must be divisible by M for IVFPQ");
#endif
#if BASELINE_INDEX_TYPE == 3
  (void)ParseSqType(cfg.sq_type);
#endif
#if BASELINE_INDEX_TYPE == 4
  const uint32_t opq_m = cfg.opq_M == 0 ? cfg.M : cfg.opq_M;
  if (cfg.M == 0) throw std::runtime_error("M must be > 0 for OPQ+IVFPQ");
  if (opq_m == 0) throw std::runtime_error("opq_M must be > 0 for OPQ+IVFPQ");
  if (cfg.nbits == 0 || cfg.nbits > 16) throw std::runtime_error("nbits must be in [1,16] for OPQ+IVFPQ");
  if (cfg.dim % cfg.M != 0) throw std::runtime_error("dim must be divisible by M for OPQ+IVFPQ");
  if (cfg.dim % opq_m != 0) throw std::runtime_error("dim must be divisible by opq_M for OPQ+IVFPQ");
#endif
#if BASELINE_INDEX_TYPE == 2
  if (cfg.nb_bits == 0 || cfg.nb_bits > 8) throw std::runtime_error("nb_bits must be in [1,8] for IVFRaBitQ");
  if (cfg.qb > 8) throw std::runtime_error("qb must be in [0,8] for IVFRaBitQ");
#endif
#if defined(BASELINE_ENABLE_LIRE)
  if (!IsL2Metric(cfg.metric)) {
    throw std::runtime_error("IndexIVFFlatLIRE supports only metric=l2");
  }
  if (cfg.lire_max_nlist < cfg.nlist) {
    throw std::runtime_error("lire_max_nlist must be >= nlist");
  }
  if (cfg.lire_split_threshold < 2) {
    throw std::runtime_error("lire_split_threshold must be >= 2");
  }
  if (cfg.lire_max_splits_per_batch == 0) {
    throw std::runtime_error("lire_max_splits_per_batch must be > 0");
  }
#endif
}

void AddRange(faiss::Index& index,
              const FvecsData& base,
              uint32_t begin,
              uint32_t end,
              RunState* state,
              const Config& cfg,
              double* update_ms) {
  if (!state || begin > end || end > base.n) throw std::runtime_error("invalid add range");
  const uint32_t rows = end - begin;
  if (rows == 0) {
    if (update_ms) *update_ms = 0.0;
    return;
  }
  std::vector<faiss::idx_t> ids(rows);
  for (uint32_t i = 0; i < rows; ++i) ids[i] = static_cast<faiss::idx_t>(begin + i);
#ifdef _OPENMP
  const int old_threads = omp_get_max_threads();
  omp_set_num_threads(static_cast<int>(AddThreadsForIndex(cfg)));
#endif
  auto t0 = std::chrono::high_resolution_clock::now();
  index.add_with_ids(static_cast<faiss::idx_t>(rows), RowPtr(base, begin), ids.data());
  auto t1 = std::chrono::high_resolution_clock::now();
#ifdef _OPENMP
  omp_set_num_threads(old_threads);
#endif
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  state->active_ids.insert(state->active_ids.end(), ids.begin(), ids.end());
  state->seen_rows = end;
  state->total_update_ms += ms;
  if (update_ms) *update_ms = ms;
}

void PrintConfig(const Config& cfg) {
  std::cout << "[CONFIG] algorithm=" << BASELINE_ALGO_NAME
            << ", metric=" << cfg.metric
            << ", topk=" << cfg.topk
            << ", nlist=" << cfg.nlist
            << ", nprobe=" << cfg.nprobe
            << ", train_rows=" << cfg.train_rows
            << ", max_queries=" << cfg.max_queries
            << ", add_threads=" << cfg.add_threads
            << ", search_threads=" << cfg.search_threads
            << ", gt_threads=" << cfg.gt_threads;
#if BASELINE_INDEX_TYPE == 1
  std::cout << ", M=" << cfg.M << ", nbits=" << cfg.nbits
            << ", rerank_factor=" << cfg.rerank_factor
            << ", exact_rerank=" << std::boolalpha << cfg.exact_rerank
            << ", unsafe_parallel_add=" << cfg.unsafe_parallel_add << std::noboolalpha;
#endif
#if BASELINE_INDEX_TYPE == 3
  std::cout << ", sq_type=" << cfg.sq_type
            << ", rerank_factor=" << cfg.rerank_factor
            << ", exact_rerank=" << std::boolalpha << cfg.exact_rerank
            << std::noboolalpha;
#endif
#if BASELINE_INDEX_TYPE == 4
  std::cout << ", M=" << cfg.M << ", nbits=" << cfg.nbits
            << ", opq_M=" << cfg.opq_M
            << ", opq_niter=" << cfg.opq_niter
            << ", opq_niter_pq=" << cfg.opq_niter_pq
            << ", opq_max_train_points=" << cfg.opq_max_train_points
            << ", rerank_factor=" << cfg.rerank_factor
            << ", exact_rerank=" << std::boolalpha << cfg.exact_rerank
            << ", unsafe_parallel_add=" << cfg.unsafe_parallel_add << std::noboolalpha;
#endif
#if BASELINE_INDEX_TYPE == 2
  std::cout << ", nb_bits=" << cfg.nb_bits
            << ", qb=" << cfg.qb
            << ", centered=" << std::boolalpha << cfg.centered
            << ", rerank_factor=" << cfg.rerank_factor
            << ", exact_rerank=" << cfg.exact_rerank << std::noboolalpha;
#endif
#if defined(BASELINE_ENABLE_LIRE)
  std::cout << ", lire_max_nlist=" << cfg.lire_max_nlist
            << ", lire_split_threshold=" << cfg.lire_split_threshold
            << ", lire_reassign_k=" << cfg.lire_reassign_k
            << ", lire_enable_reassign=" << std::boolalpha
            << cfg.lire_enable_reassign
            << ", lire_validate_invariants=" << cfg.lire_validate_invariants
            << std::noboolalpha;
#endif
  std::cout << std::endl;
}

void PrintEvalLine(const Snapshot& s) {
#if BASELINE_MODE_MS
  std::cout << "[EVAL] #" << s.search_id
            << " op_id=" << s.op_id
            << " active_rows=" << s.active_rows
            << " recall=" << std::fixed << std::setprecision(6) << s.metrics.recall
            << " recall_new=" << s.metrics.age.recall_new
            << " recall_old=" << s.metrics.age.recall_old
            << " qps=" << s.metrics.qps
            << " avg_query_ms=" << s.metrics.avg_query_ms
            << " avg_scanned=" << s.metrics.avg_scanned
            << " cumulative_update_ms=" << s.cumulative_update_ms
            << std::endl;
#else
  std::cout << "[SNAPSHOT] snapshot_size=" << s.base_rows
            << " recall=" << std::fixed << std::setprecision(6) << s.metrics.recall
            << " recall_new=" << s.metrics.age.recall_new
            << " recall_old=" << s.metrics.age.recall_old
            << " latency_ms=" << s.metrics.avg_query_ms
            << " qps=" << s.metrics.qps
            << " update_ms=" << s.update_ms
            << " throughput=" << s.update_throughput_vecps
            << std::endl;
#endif
}

void WriteSummaryJson(const std::string& path,
                      const Config& cfg,
                      const std::string& config_path,
                      const std::string& base_path,
                      const std::string& query_path,
                      const std::optional<Workload>& workload,
                      const std::vector<Snapshot>& snapshots,
                      const EvalMetrics& final_metrics,
                      double train_ms,
                      double total_update_ms) {
  fs::path p(path);
  if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
  std::ofstream ofs(path);
  if (!ofs) throw std::runtime_error("Failed to open JSON output: " + path);
  ofs << std::fixed << std::setprecision(6);

  std::vector<double> recall_values;
  std::vector<double> qps_values;
  std::vector<double> latency_values;
  std::vector<double> e2e_latency_values;
  std::vector<double> update_tp_values;
  for (const auto& s : snapshots) {
    if (s.skipped_warmup) continue;
    recall_values.push_back(s.metrics.recall);
    qps_values.push_back(s.metrics.qps);
    latency_values.push_back(s.metrics.avg_query_ms);
    e2e_latency_values.push_back(s.metrics.query_eval_ms + s.snapshot_maintenance_ms);
    update_tp_values.push_back(s.update_throughput_vecps);
  }
  if (recall_values.empty() && !snapshots.empty()) {
    for (const auto& s : snapshots) {
      recall_values.push_back(s.metrics.recall);
      qps_values.push_back(s.metrics.qps);
      latency_values.push_back(s.metrics.avg_query_ms);
      e2e_latency_values.push_back(s.metrics.query_eval_ms + s.snapshot_maintenance_ms);
      update_tp_values.push_back(s.update_throughput_vecps);
    }
  }
  const auto recall_summary = SummarizeSeries(recall_values);
  const auto qps_summary = SummarizeSeries(qps_values);
  const auto latency_summary = SummarizeSeries(latency_values);
  const auto e2e_latency_summary = SummarizeSeries(e2e_latency_values);
  const auto update_tp_summary = SummarizeSeries(update_tp_values);

  uint64_t gt_new_total = 0;
  uint64_t gt_old_total = 0;
  uint64_t hit_new_total = 0;
  uint64_t hit_old_total = 0;
  for (const auto& s : snapshots) {
    if (s.skipped_warmup) continue;
    gt_new_total += s.metrics.age.gt_new_total;
    gt_old_total += s.metrics.age.gt_old_total;
    hit_new_total += s.metrics.age.hit_new_total;
    hit_old_total += s.metrics.age.hit_old_total;
  }
  if (gt_new_total + gt_old_total == 0) {
    gt_new_total = final_metrics.age.gt_new_total;
    gt_old_total = final_metrics.age.gt_old_total;
    hit_new_total = final_metrics.age.hit_new_total;
    hit_old_total = final_metrics.age.hit_old_total;
  }
  const uint64_t gt_total = gt_new_total + gt_old_total;
  const double recall_new = gt_new_total > 0 ? static_cast<double>(hit_new_total) / gt_new_total : 0.0;
  const double recall_old = gt_old_total > 0 ? static_cast<double>(hit_old_total) / gt_old_total : 0.0;
  const double gt_new_ratio = gt_total > 0 ? static_cast<double>(gt_new_total) / gt_total : 0.0;

  const uint64_t rss_bytes = ReadProcStatusBytes("VmRSS:");
  const uint64_t hwm_bytes = ReadProcStatusBytes("VmHWM:");
  const double amortized_update_tp = total_update_ms > 0.0 && !snapshots.empty()
      ? static_cast<double>(snapshots.back().active_rows) / (total_update_ms / 1000.0)
      : 0.0;

  ofs << "{\n";
  ofs << "  \"engine\": \"faiss\",\n";
  ofs << "  \"baseline\": \"" << BASELINE_ALGO_NAME << "\",\n";
  ofs << "  \"index_type\": \"" << BASELINE_ALGO_NAME << "\",\n";
  ofs << "  \"dataset\": \"" << JsonEscape(base_path) << "\",\n";
  ofs << "  \"query\": \"" << JsonEscape(query_path) << "\",\n";
  ofs << "  \"config_path\": \"" << JsonEscape(config_path) << "\",\n";
  if (workload) {
    ofs << "  \"workload\": \"" << JsonEscape(workload->path) << "\",\n";
  }
  ofs << "  \"final_metrics\": {\n";
  ofs << "    \"recall@" << cfg.topk << "\": " << final_metrics.recall << ",\n";
  ofs << "    \"recall\": " << final_metrics.recall << ",\n";
#if BASELINE_INDEX_TYPE == 1
  ofs << "    \"adc_recall\": " << final_metrics.raw_recall << ",\n";
#endif
#if BASELINE_INDEX_TYPE == 3
  ofs << "    \"sq_recall\": " << final_metrics.raw_recall << ",\n";
#endif
#if BASELINE_INDEX_TYPE == 4
  ofs << "    \"adc_recall\": " << final_metrics.raw_recall << ",\n";
#endif
#if BASELINE_INDEX_TYPE == 2
  ofs << "    \"rabitq_recall\": " << final_metrics.raw_recall << ",\n";
#endif
  ofs << "    \"recall_new\": " << final_metrics.age.recall_new << ",\n";
  ofs << "    \"recall_old\": " << final_metrics.age.recall_old << ",\n";
  ofs << "    \"gt_new_ratio\": " << final_metrics.age.gt_new_ratio << ",\n";
  ofs << "    \"gt_new_total\": " << final_metrics.age.gt_new_total << ",\n";
  ofs << "    \"gt_old_total\": " << final_metrics.age.gt_old_total << ",\n";
  ofs << "    \"hit_new_total\": " << final_metrics.age.hit_new_total << ",\n";
  ofs << "    \"hit_old_total\": " << final_metrics.age.hit_old_total << ",\n";
  ofs << "    \"qps\": " << final_metrics.qps << ",\n";
  ofs << "    \"avg_query_ms\": " << final_metrics.avg_query_ms << ",\n";
  ofs << "    \"avg_scanned\": " << final_metrics.avg_scanned << "\n";
  ofs << "  },\n";

  ofs << "  \"summary\": {\n";
  ofs << "    \"recall_avg\": " << recall_summary.avg << ",\n";
  ofs << "    \"recall_p5\": " << recall_summary.p5 << ",\n";
  ofs << "    \"recall_min\": " << recall_summary.min << ",\n";
  ofs << "    \"recall_new\": " << recall_new << ",\n";
  ofs << "    \"recall_old\": " << recall_old << ",\n";
  ofs << "    \"gt_new_ratio\": " << gt_new_ratio << ",\n";
  ofs << "    \"gt_new_total\": " << gt_new_total << ",\n";
  ofs << "    \"gt_old_total\": " << gt_old_total << ",\n";
  ofs << "    \"hit_new_total\": " << hit_new_total << ",\n";
  ofs << "    \"hit_old_total\": " << hit_old_total << ",\n";
  ofs << "    \"qps_avg\": " << qps_summary.avg << ",\n";
  ofs << "    \"qps_p5\": " << qps_summary.p5 << ",\n";
  ofs << "    \"latency_avg_ms\": " << latency_summary.avg << ",\n";
  ofs << "    \"latency_p95_ms\": " << latency_summary.p95 << ",\n";
  ofs << "    \"latency_p99_ms\": " << latency_summary.p99 << ",\n";
  ofs << "    \"e2e_latency_avg_ms\": " << e2e_latency_summary.avg << ",\n";
  ofs << "    \"update_throughput_avg_vecps\": " << update_tp_summary.avg << ",\n";
  ofs << "    \"amortized_update_throughput_avg_vecps\": " << amortized_update_tp << ",\n";
  ofs << "    \"train_ms\": " << train_ms << ",\n";
  ofs << "    \"update_ms\": " << total_update_ms << ",\n";
  ofs << "    \"memory_rss_bytes\": " << rss_bytes << ",\n";
  ofs << "    \"memory_hwm_bytes\": " << hwm_bytes << "\n";
  ofs << "  },\n";

  ofs << "  \"params\": {\n";
  ofs << "    \"metric\": \"" << JsonEscape(cfg.metric) << "\",\n";
  ofs << "    \"topk\": " << cfg.topk << ",\n";
  ofs << "    \"nlist\": " << cfg.nlist << ",\n";
  ofs << "    \"nprobe\": " << cfg.nprobe << ",\n";
  ofs << "    \"main_index_rows\": " << cfg.main_index_rows << ",\n";
  ofs << "    \"snapshot_interval\": " << cfg.snapshot_interval << ",\n";
  ofs << "    \"train_rows\": " << cfg.train_rows << ",\n";
  ofs << "    \"max_queries\": " << cfg.max_queries << ",\n";
  ofs << "    \"add_threads\": " << cfg.add_threads << ",\n";
  ofs << "    \"search_threads\": " << cfg.search_threads << ",\n";
  ofs << "    \"gt_threads\": " << cfg.gt_threads;
#if defined(BASELINE_ENABLE_LIRE)
  ofs << ",\n    \"lire_max_nlist\": " << cfg.lire_max_nlist << ",\n";
  ofs << "    \"lire_split_threshold\": " << cfg.lire_split_threshold << ",\n";
  ofs << "    \"lire_reassign_k\": " << cfg.lire_reassign_k << ",\n";
  ofs << "    \"lire_max_splits_per_batch\": "
      << cfg.lire_max_splits_per_batch << ",\n";
  ofs << "    \"lire_enable_reassign\": "
      << (cfg.lire_enable_reassign ? "true" : "false") << ",\n";
  ofs << "    \"lire_validate_invariants\": "
      << (cfg.lire_validate_invariants ? "true" : "false");
#endif
#if BASELINE_INDEX_TYPE == 1
  ofs << ",\n    \"M\": " << cfg.M << ",\n";
  ofs << "    \"nbits\": " << cfg.nbits << ",\n";
  ofs << "    \"rerank_factor\": " << cfg.rerank_factor << ",\n";
  ofs << "    \"exact_rerank\": " << (cfg.exact_rerank ? "true" : "false") << ",\n";
  ofs << "    \"unsafe_parallel_add\": " << (cfg.unsafe_parallel_add ? "true" : "false");
#endif
#if BASELINE_INDEX_TYPE == 3
  ofs << ",\n    \"sq_type\": \"" << JsonEscape(cfg.sq_type) << "\",\n";
  ofs << "    \"rerank_factor\": " << cfg.rerank_factor << ",\n";
  ofs << "    \"exact_rerank\": " << (cfg.exact_rerank ? "true" : "false");
#endif
#if BASELINE_INDEX_TYPE == 4
  ofs << ",\n    \"M\": " << cfg.M << ",\n";
  ofs << "    \"nbits\": " << cfg.nbits << ",\n";
  ofs << "    \"opq_M\": " << cfg.opq_M << ",\n";
  ofs << "    \"opq_niter\": " << cfg.opq_niter << ",\n";
  ofs << "    \"opq_niter_pq\": " << cfg.opq_niter_pq << ",\n";
  ofs << "    \"opq_niter_pq_0\": " << cfg.opq_niter_pq_0 << ",\n";
  ofs << "    \"opq_max_train_points\": " << cfg.opq_max_train_points << ",\n";
  ofs << "    \"rerank_factor\": " << cfg.rerank_factor << ",\n";
  ofs << "    \"exact_rerank\": " << (cfg.exact_rerank ? "true" : "false") << ",\n";
  ofs << "    \"unsafe_parallel_add\": " << (cfg.unsafe_parallel_add ? "true" : "false");
#endif
#if BASELINE_INDEX_TYPE == 2
  ofs << ",\n    \"nb_bits\": " << cfg.nb_bits << ",\n";
  ofs << "    \"qb\": " << cfg.qb << ",\n";
  ofs << "    \"centered\": " << (cfg.centered ? "true" : "false") << ",\n";
  ofs << "    \"rerank_factor\": " << cfg.rerank_factor << ",\n";
  ofs << "    \"exact_rerank\": " << (cfg.exact_rerank ? "true" : "false");
#endif
  ofs << "\n  },\n";

  ofs << "  \"snapshots\": [\n";
  for (size_t i = 0; i < snapshots.size(); ++i) {
    const auto& s = snapshots[i];
#if BASELINE_MODE_MS
    ofs << "    {\"search_id\": " << s.search_id
        << ", \"op_id\": " << s.op_id
        << ", \"active_rows\": " << s.active_rows
        << ", \"main_rows\": " << s.main_rows
        << ", \"active_delta_rows\": " << s.active_delta_rows
        << ", \"frozen_delta_rows\": " << s.frozen_delta_rows
        << ", \"round\": " << s.round
        << ", \"cluster\": " << s.cluster
        << ", \"recall\": " << s.metrics.recall
        << ", \"recall_new\": " << s.metrics.age.recall_new
        << ", \"recall_old\": " << s.metrics.age.recall_old
        << ", \"gt_new_ratio\": " << s.metrics.age.gt_new_ratio
        << ", \"gt_new_total\": " << s.metrics.age.gt_new_total
        << ", \"gt_old_total\": " << s.metrics.age.gt_old_total
        << ", \"hit_new_total\": " << s.metrics.age.hit_new_total
        << ", \"hit_old_total\": " << s.metrics.age.hit_old_total
        << ", \"avg_query_ms\": " << s.metrics.avg_query_ms
        << ", \"p50_query_ms\": " << s.metrics.p50_query_ms
        << ", \"p99_query_ms\": " << s.metrics.p99_query_ms
        << ", \"qps\": " << s.metrics.qps
        << ", \"avg_scanned\": " << s.metrics.avg_scanned
        << ", \"cumulative_update_ms\": " << s.cumulative_update_ms
        << ", \"snapshot_rows\": " << s.snapshot_rows
        << ", \"snapshot_maintenance_ms\": " << s.snapshot_maintenance_ms
        << ", \"update_throughput_vecps\": " << s.update_throughput_vecps
        << ", \"amortized_update_throughput_vecps\": " << s.amortized_update_throughput_vecps
        << ", \"skipped_warmup\": " << (s.skipped_warmup ? "true" : "false")
        << "}";
#else
    ofs << "    {\n";
    ofs << "      \"snapshot_size\": " << s.base_rows << ",\n";
    ofs << "      \"recall@" << cfg.topk << "\": " << s.metrics.recall << ",\n";
    ofs << "      \"recall\": " << s.metrics.recall << ",\n";
    ofs << "      \"recall_new\": " << s.metrics.age.recall_new << ",\n";
    ofs << "      \"recall_old\": " << s.metrics.age.recall_old << ",\n";
    ofs << "      \"gt_new_ratio\": " << s.metrics.age.gt_new_ratio << ",\n";
    ofs << "      \"gt_new_total\": " << s.metrics.age.gt_new_total << ",\n";
    ofs << "      \"gt_old_total\": " << s.metrics.age.gt_old_total << ",\n";
    ofs << "      \"hit_new_total\": " << s.metrics.age.hit_new_total << ",\n";
    ofs << "      \"hit_old_total\": " << s.metrics.age.hit_old_total << ",\n";
#if BASELINE_INDEX_TYPE == 1
    ofs << "      \"adc_recall\": " << s.metrics.raw_recall << ",\n";
#endif
#if BASELINE_INDEX_TYPE == 3
    ofs << "      \"sq_recall\": " << s.metrics.raw_recall << ",\n";
#endif
#if BASELINE_INDEX_TYPE == 4
    ofs << "      \"adc_recall\": " << s.metrics.raw_recall << ",\n";
#endif
#if BASELINE_INDEX_TYPE == 2
    ofs << "      \"rabitq_recall\": " << s.metrics.raw_recall << ",\n";
#endif
    ofs << "      \"rerank_candidates\": " << s.metrics.rerank_candidates << ",\n";
    ofs << "      \"latency_ms\": " << s.metrics.avg_query_ms << ",\n";
    ofs << "      \"query_qps\": " << s.metrics.qps << ",\n";
    ofs << "      \"avg_scanned\": " << s.metrics.avg_scanned << ",\n";
    ofs << "      \"update_ms\": " << s.update_ms << ",\n";
    ofs << "      \"snapshot_total_ms\": " << s.snapshot_total_ms << ",\n";
    ofs << "      \"throughput\": " << s.update_throughput_vecps << ",\n";
    ofs << "      \"amortized_update_throughput_vecps\": " << s.amortized_update_throughput_vecps << "\n";
    ofs << "    }";
#endif
    if (i + 1 < snapshots.size()) ofs << ",";
    ofs << "\n";
  }
  ofs << "  ]\n";
  ofs << "}\n";
}

int RunEvalMain(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <config.json> <dataset_dir|base.fvecs> [online_eval.json]\n";
    return 1;
  }
  const std::string config_path = argv[1];
  const std::string dataset_arg = argv[2];
  std::string output_path = argc >= 4 ? argv[3] : "";

  Config cfg = LoadConfigJson(config_path);
  std::string base_path = ResolveFileWithSuffix(dataset_arg, "_base.fvecs");
  std::string query_path = ResolveFileWithSuffix(fs::path(base_path).parent_path().string(), "_query.fvecs");
  if (output_path.empty()) output_path = BuildDefaultResultPath(config_path, base_path).string();

  FvecsData base = LoadFvecs(base_path);
  FvecsData query = LoadFvecs(query_path);
  if (base.d != query.d) throw std::runtime_error("Base/query dim mismatch");
  cfg.dim = base.d;
  if (BASELINE_INDEX_TYPE == 4 && cfg.opq_M == 0) cfg.opq_M = cfg.M;
  LimitQueries(&query, cfg.max_queries);
  ValidateConfig(cfg, base);
  if (cfg.train_rows == 0) cfg.train_rows = cfg.main_index_rows;
  cfg.train_rows = std::min<uint32_t>(cfg.train_rows, base.n);

  if (IsCosineMetric(cfg.metric)) {
    std::cout << "[METRIC] cosine requested; L2-normalizing base/query and using normalized L2 ranking" << std::endl;
    NormalizeRowsL2(&base);
    NormalizeRowsL2(&query);
  }

#ifdef _OPENMP
  omp_set_dynamic(0);
#endif

  std::cout << "[INFO] base=" << base_path << " n=" << base.n << " d=" << base.d << std::endl;
  std::cout << "[INFO] query=" << query_path << " n=" << query.n << std::endl;
  std::cout << "[INFO] output=" << output_path << std::endl;
  PrintConfig(cfg);

  faiss::IndexFlatL2 quantizer(static_cast<int>(cfg.dim));
  auto index = CreateConfiguredIndex(cfg, &quantizer);
  SetIndexParams(*index, cfg);

#ifdef _OPENMP
  int old_threads = omp_get_max_threads();
  omp_set_num_threads(static_cast<int>(cfg.add_threads));
#endif
  std::cout << "[BUILD] training " << BASELINE_ALGO_NAME << " on first " << cfg.train_rows << " base vectors" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  index->train(static_cast<faiss::idx_t>(cfg.train_rows), base.values.data());
  auto t1 = std::chrono::high_resolution_clock::now();
#ifdef _OPENMP
  omp_set_num_threads(old_threads);
#endif
  const double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[BUILD] train_ms=" << train_ms << std::endl;

  RunState state;
  state.active_ids.reserve(base.n);
  std::vector<Snapshot> snapshots;
  snapshots.reserve(1 + (base.n > cfg.main_index_rows
      ? (base.n - cfg.main_index_rows + cfg.snapshot_interval - 1) / cfg.snapshot_interval
      : 0));

  double init_update_ms = 0.0;
  std::cout << "[INSERT] initial base[0:" << cfg.main_index_rows << ")" << std::endl;
  AddRange(*index, base, 0, cfg.main_index_rows, &state, cfg, &init_update_ms);

  uint32_t last_begin = 0;
  uint32_t last_end = cfg.main_index_rows;
  Snapshot first;
  first.search_id = 1;
  first.active_rows = state.seen_rows;
  first.main_rows = cfg.main_index_rows;
  first.base_rows = state.seen_rows;
  first.snapshot_rows = cfg.main_index_rows;
  first.update_ms = init_update_ms;
  first.cumulative_update_ms = state.total_update_ms;
  first.snapshot_maintenance_ms = init_update_ms;
  first.metrics = EvaluateSnapshot(*index, base, query, state.seen_rows, last_begin, last_end, cfg);
  first.snapshot_total_ms = init_update_ms + first.metrics.query_eval_ms;
  first.update_throughput_vecps = init_update_ms > 0.0 ? static_cast<double>(cfg.main_index_rows) / (init_update_ms / 1000.0) : 0.0;
  first.amortized_update_throughput_vecps = state.total_update_ms > 0.0 ? static_cast<double>(state.seen_rows) / (state.total_update_ms / 1000.0) : 0.0;
  snapshots.push_back(first);
  PrintEvalLine(first);
  WriteSummaryJson(output_path, cfg, config_path, base_path, query_path, std::nullopt, snapshots, first.metrics, train_ms, state.total_update_ms);

  uint32_t search_id = 1;
  while (state.seen_rows < base.n) {
    const uint32_t begin = state.seen_rows;
    const uint32_t end = std::min<uint32_t>(base.n, begin + cfg.snapshot_interval);
    const uint32_t rows = end - begin;
    std::cout << "[INSERT] base[" << begin << ":" << end << ")" << std::endl;
    double update_ms = 0.0;
    AddRange(*index, base, begin, end, &state, cfg, &update_ms);
    search_id++;
    Snapshot snap;
    snap.search_id = search_id;
    snap.active_rows = state.seen_rows;
    snap.main_rows = cfg.main_index_rows;
    snap.base_rows = state.seen_rows;
    snap.snapshot_rows = rows;
    snap.update_ms = update_ms;
    snap.cumulative_update_ms = state.total_update_ms;
    snap.snapshot_maintenance_ms = update_ms;
    snap.metrics = EvaluateSnapshot(*index, base, query, state.seen_rows, begin, end, cfg);
    snap.snapshot_total_ms = update_ms + snap.metrics.query_eval_ms;
    snap.update_throughput_vecps = update_ms > 0.0 ? static_cast<double>(rows) / (update_ms / 1000.0) : 0.0;
    snap.amortized_update_throughput_vecps = state.total_update_ms > 0.0 ? static_cast<double>(state.seen_rows) / (state.total_update_ms / 1000.0) : 0.0;
    snapshots.push_back(snap);
    PrintEvalLine(snap);
    WriteSummaryJson(output_path, cfg, config_path, base_path, query_path, std::nullopt, snapshots, snap.metrics, train_ms, state.total_update_ms);
  }

  const EvalMetrics final_metrics = snapshots.empty() ? EvalMetrics{} : snapshots.back().metrics;
  WriteSummaryJson(output_path, cfg, config_path, base_path, query_path, std::nullopt, snapshots, final_metrics, train_ms, state.total_update_ms);
  std::cout << "[DONE] wrote JSON to " << output_path << std::endl;
  return 0;
}

bool LooksLikeWorkloadPath(const std::string& p) {
  return EndsWith(p, ".json") && p.find("workload") != std::string::npos;
}

int RunEvalMsMain(int argc, char** argv) {
  if (argc < 3 || argc > 6) {
    std::cerr << "Usage: " << argv[0] << " <config.json> <dataset_dir|base.fvecs> [query_or_dir] [workload.json] [online_eval.json]\n";
    return 1;
  }
  const std::string config_path = argv[1];
  const std::string dataset_arg = argv[2];
  std::string query_arg;
  std::string workload_path;
  std::string output_path;
  if (argc == 4) {
    if (LooksLikeWorkloadPath(argv[3])) workload_path = argv[3];
    else if (EndsWith(argv[3], ".json")) output_path = argv[3];
    else query_arg = argv[3];
  } else if (argc == 5) {
    query_arg = argv[3];
    if (LooksLikeWorkloadPath(argv[4])) workload_path = argv[4];
    else output_path = argv[4];
  } else if (argc == 6) {
    query_arg = argv[3];
    workload_path = argv[4];
    output_path = argv[5];
  }

  Config cfg = LoadConfigJson(config_path);
  std::string base_path = ResolveFileWithSuffix(dataset_arg, "_base.fvecs");
  std::string query_path = query_arg.empty()
      ? ResolveFileWithSuffix(fs::path(base_path).parent_path().string(), "_query.fvecs")
      : ResolveFileWithSuffix(query_arg, "_query.fvecs");
  if (workload_path.empty()) workload_path = ResolveFileWithSuffix(dataset_arg, "_workload.json");
  Workload workload = LoadWorkloadJson(workload_path);
  if (output_path.empty()) output_path = BuildDefaultResultPath(config_path, base_path).string();

  FvecsData base = LoadFvecs(base_path);
  FvecsData query = LoadFvecs(query_path);
  if (base.d != query.d) throw std::runtime_error("Base/query dim mismatch");
  cfg.dim = base.d;
  if (BASELINE_INDEX_TYPE == 4 && cfg.opq_M == 0) cfg.opq_M = cfg.M;
  LimitQueries(&query, cfg.max_queries);
  ValidateConfig(cfg, base);
  if (cfg.train_rows == 0) cfg.train_rows = cfg.main_index_rows > 0 ? cfg.main_index_rows : std::min<uint32_t>(base.n, std::max<uint32_t>(cfg.nlist, 1));
  cfg.train_rows = std::min<uint32_t>(cfg.train_rows, base.n);

  if (IsCosineMetric(cfg.metric)) {
    std::cout << "[METRIC] cosine requested; L2-normalizing base/query and using normalized L2 ranking" << std::endl;
    NormalizeRowsL2(&base);
    NormalizeRowsL2(&query);
  }

#ifdef _OPENMP
  omp_set_dynamic(0);
#endif

  std::cout << "[INFO] base=" << base_path << " n=" << base.n << " d=" << base.d << std::endl;
  std::cout << "[INFO] query=" << query_path << " n=" << query.n << std::endl;
  std::cout << "[WORKLOAD] " << workload.path << " ops=" << workload.operations.size() << std::endl;
  std::cout << "[INFO] output=" << output_path << std::endl;
  PrintConfig(cfg);

  faiss::IndexFlatL2 quantizer(static_cast<int>(cfg.dim));
  auto index = CreateConfiguredIndex(cfg, &quantizer);
  SetIndexParams(*index, cfg);

#ifdef _OPENMP
  int old_threads = omp_get_max_threads();
  omp_set_num_threads(static_cast<int>(cfg.add_threads));
#endif
  std::cout << "[BUILD] training " << BASELINE_ALGO_NAME << " on first " << cfg.train_rows << " base vectors" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  index->train(static_cast<faiss::idx_t>(cfg.train_rows), base.values.data());
  auto t1 = std::chrono::high_resolution_clock::now();
#ifdef _OPENMP
  omp_set_num_threads(old_threads);
#endif
  const double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[BUILD] train_ms=" << train_ms << std::endl;

  RunState state;
  state.active_ids.reserve(base.n);
  std::vector<Snapshot> snapshots;
  snapshots.reserve(workload.operations.size());
  uint32_t search_id = 0;
  uint32_t last_insert_begin = 0;
  uint32_t last_insert_end = 0;
  double pending_update_ms = 0.0;
  uint32_t pending_insert_rows = 0;

  for (const auto& op : workload.operations) {
    if (op.operation == "insert") {
      if (op.end > base.n || op.start > op.end) {
        throw std::runtime_error("invalid insert op range at op_id=" + std::to_string(op.op_id));
      }
      if (op.start != state.seen_rows) {
        throw std::runtime_error("workload is not prefix-contiguous at op_id=" + std::to_string(op.op_id));
      }
      std::cout << "[INSERT] op_id=" << op.op_id << " base[" << op.start << ":" << op.end << ")" << std::endl;
      double update_ms = 0.0;
      AddRange(*index, base, op.start, op.end, &state, cfg, &update_ms);
      last_insert_begin = op.start;
      last_insert_end = op.end;
      pending_update_ms += update_ms;
      pending_insert_rows += (op.end - op.start);
      std::cout << "[INSERT] done rows=" << (op.end - op.start)
                << " update_ms=" << update_ms
                << " active_rows=" << state.seen_rows << std::endl;
    } else if (op.operation == "search") {
      search_id++;
      Snapshot snap;
      snap.search_id = search_id;
      snap.op_id = op.op_id;
      snap.active_rows = state.seen_rows;
      snap.main_rows = state.seen_rows;
      snap.base_rows = state.seen_rows;
      snap.round = op.round;
      snap.cluster = op.cluster;
      snap.cumulative_update_ms = state.total_update_ms;
      snap.snapshot_rows = pending_insert_rows;
      snap.snapshot_maintenance_ms = pending_update_ms;
      if (state.seen_rows == 0) {
        snap.skipped_warmup = true;
        snapshots.push_back(snap);
        continue;
      }
      snap.metrics = EvaluateSnapshot(*index, base, query, state.seen_rows,
                                      last_insert_begin, last_insert_end, cfg);
      snap.snapshot_total_ms = snap.metrics.query_eval_ms + snap.snapshot_maintenance_ms;
      snap.update_throughput_vecps = snap.snapshot_rows > 0 && snap.snapshot_maintenance_ms > 0.0
          ? static_cast<double>(snap.snapshot_rows) / (snap.snapshot_maintenance_ms / 1000.0)
          : 0.0;
      snap.amortized_update_throughput_vecps = state.total_update_ms > 0.0
          ? static_cast<double>(state.seen_rows) / (state.total_update_ms / 1000.0)
          : 0.0;
      snapshots.push_back(snap);
      PrintEvalLine(snap);
      WriteSummaryJson(output_path, cfg, config_path, base_path, query_path, workload, snapshots,
                       snap.metrics, train_ms, state.total_update_ms);
      pending_update_ms = 0.0;
      pending_insert_rows = 0;
    } else {
      throw std::runtime_error("unsupported workload operation: " + op.operation);
    }
  }

  if (snapshots.empty() || snapshots.back().active_rows != state.seen_rows) {
    search_id++;
    Snapshot snap;
    snap.search_id = search_id;
    snap.op_id = workload.operations.empty() ? 0 : workload.operations.back().op_id + 1;
    snap.active_rows = state.seen_rows;
    snap.main_rows = state.seen_rows;
    snap.base_rows = state.seen_rows;
    snap.cumulative_update_ms = state.total_update_ms;
    snap.snapshot_rows = pending_insert_rows;
    snap.snapshot_maintenance_ms = pending_update_ms;
    if (state.seen_rows > 0) {
      snap.metrics = EvaluateSnapshot(*index, base, query, state.seen_rows,
                                      last_insert_begin, last_insert_end, cfg);
      snap.snapshot_total_ms = snap.metrics.query_eval_ms + snap.snapshot_maintenance_ms;
      snap.update_throughput_vecps = snap.snapshot_rows > 0 && snap.snapshot_maintenance_ms > 0.0
          ? static_cast<double>(snap.snapshot_rows) / (snap.snapshot_maintenance_ms / 1000.0)
          : 0.0;
      snap.amortized_update_throughput_vecps = state.total_update_ms > 0.0
          ? static_cast<double>(state.seen_rows) / (state.total_update_ms / 1000.0)
          : 0.0;
    } else {
      snap.skipped_warmup = true;
    }
    snapshots.push_back(snap);
    if (!snap.skipped_warmup) PrintEvalLine(snap);
  }

  const EvalMetrics final_metrics = snapshots.empty() ? EvalMetrics{} : snapshots.back().metrics;
  WriteSummaryJson(output_path, cfg, config_path, base_path, query_path, workload, snapshots,
                   final_metrics, train_ms, state.total_update_ms);
  std::cout << "[DONE] algorithm=" << BASELINE_ALGO_NAME
            << " searches=" << search_id
            << " snapshots=" << snapshots.size()
            << " active_rows=" << state.seen_rows
            << " summary=" << output_path << std::endl;
  return 0;
}

}  // namespace baseline_eval

#ifndef BASELINE_SUPPRESS_MAIN
int main(int argc, char** argv) {
  try {
#if BASELINE_MODE_MS
    return baseline_eval::RunEvalMsMain(argc, argv);
#else
    return baseline_eval::RunEvalMain(argc, argv);
#endif
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
#endif
