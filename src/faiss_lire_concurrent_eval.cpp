#define BASELINE_ALGO_NAME "IndexIVFFlatLIREConcurrent"
#define BASELINE_INDEX_TYPE 0
#define BASELINE_MODE_MS 0
#define BASELINE_ENABLE_LIRE 1
#define BASELINE_SUPPRESS_MAIN 1
#include "faiss_baseline_run_eval_common.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <thread>
#include <unordered_set>

namespace baseline_eval {
namespace {

struct GroundTruth {
  std::vector<uint32_t> frontiers;
  std::vector<std::vector<std::vector<uint32_t>>> ids;
};

template <typename T>
bool ReadNative(std::istream& in, T* value) {
  return static_cast<bool>(in.read(reinterpret_cast<char*>(value), sizeof(T)));
}

std::string ReadArtifactString(std::istream& in) {
  uint32_t size = 0;
  if (!ReadNative(in, &size) || size > (1u << 20))
    throw std::runtime_error("Invalid or truncated GT string length");
  std::string value(size, '\0');
  if (!in.read(value.data(), size)) throw std::runtime_error("Truncated GT string");
  return value;
}

GroundTruth LoadGroundTruth(const std::string& path,
                            const std::string& base_identity,
                            const std::string& query_identity,
                            uint32_t query_count,
                            uint32_t topk,
                            uint32_t initial_rows,
                            uint32_t batch_size,
                            const std::vector<uint32_t>& expected_frontiers) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Unable to open DiSPQ GT artifact: " + path);
  char magic[8]{};
  uint32_t nq = 0, k = 0, initial = 0, batch = 0, versions = 0;
  if (!in.read(magic, sizeof(magic)) || std::string(magic, sizeof(magic)) != "DISPGT01")
    throw std::runtime_error("Unsupported DiSPQ GT magic/version");
  const std::string base = ReadArtifactString(in);
  const std::string query = ReadArtifactString(in);
  if (!ReadNative(in, &nq) || !ReadNative(in, &k) || !ReadNative(in, &initial) ||
      !ReadNative(in, &batch) || !ReadNative(in, &versions))
    throw std::runtime_error("Truncated DiSPQ GT metadata");
  if (base != base_identity || query != query_identity || nq != query_count ||
      k != topk || initial != initial_rows || batch != batch_size ||
      versions != expected_frontiers.size()) {
    throw std::runtime_error("DiSPQ GT metadata mismatch (identity/query/topk/initial/batch/frontier count)");
  }
  GroundTruth gt;
  gt.frontiers.reserve(versions);
  gt.ids.resize(versions, std::vector<std::vector<uint32_t>>(query_count));
  for (uint32_t version = 0; version < versions; ++version) {
    uint32_t frontier = 0;
    if (!ReadNative(in, &frontier) || frontier != expected_frontiers[version])
      throw std::runtime_error("DiSPQ GT frontier mismatch at version " +
                               std::to_string(version));
    gt.frontiers.push_back(frontier);
    for (uint32_t qi = 0; qi < query_count; ++qi) {
      auto& row = gt.ids[version][qi];
      row.resize(topk);
      if (!in.read(reinterpret_cast<char*>(row.data()),
                   static_cast<std::streamsize>(topk * sizeof(uint32_t))))
        throw std::runtime_error("Truncated DiSPQ GT DocId payload");
    }
  }
  char trailing = 0;
  if (in.read(&trailing, 1)) throw std::runtime_error("Unexpected trailing bytes in DiSPQ GT artifact");
  return gt;
}

uint64_t ParseUint64(const std::string& text, const std::string& option) {
  uint64_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::runtime_error("Invalid integer for " + option + ": " + text);
  return value;
}

struct QueryRecord {
  uint32_t query_id{0};
  uint32_t frontier{0};
  uint32_t gt_version{0};
  double latency_ms{0.0};
  double completion_ms{0.0};
  double recall{0.0};
  std::vector<faiss::idx_t> returned_topk;
};

struct InsertRecord {
  uint32_t batch{0};
  uint32_t begin_row{0};
  uint32_t end_row{0};
  double start_ms{0.0};
  double end_ms{0.0};
  double latency_ms{0.0};
  double throughput{0.0};
};

struct QueryWorkerJoiner {
  std::atomic<bool>& stop;
  std::vector<std::thread>& workers;
  ~QueryWorkerJoiner() {
    stop.store(true, std::memory_order_relaxed);
    for (std::thread& worker : workers)
      if (worker.joinable()) worker.join();
  }
};

double Percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(std::ceil(p * values.size())) - 1;
  return values[std::min(index, values.size() - 1)];
}

double ProgressForFrontier(uint32_t frontier,
                           uint32_t initial,
                           uint32_t final_rows,
                           uint32_t first_batch_end) {
  if (final_rows <= initial) return 100.0;
  if (frontier >= final_rows) return 100.0;
  if (frontier <= first_batch_end) return 0.0;
  const uint32_t span = final_rows - first_batch_end;
  return span == 0 ? 100.0 :
      100.0 * static_cast<double>(std::min(frontier, final_rows) - first_batch_end) / span;
}

void WriteOutputs(const std::string& summary_path,
                  const std::string& config_path,
                  const std::string& base_path,
                  const std::string& query_path,
                  const std::string& gt_path,
                  uint32_t topk,
                  uint32_t query_workers,
                  uint32_t insert_workers,
                  uint32_t maintenance_workers,
                  uint32_t initial_rows,
                  uint32_t final_rows,
                  uint32_t first_batch_end,
                  const std::vector<uint32_t>& expected_frontiers,
                  bool insert_cap_enabled,
                  uint64_t max_insert_vectors,
                  uint32_t query_count,
                  const std::vector<QueryRecord>& queries,
                  const std::vector<InsertRecord>& inserts,
                  const std::vector<LireMaintenanceEvent>& events,
                  const IndexIVFFlatLIRE& index,
                  double total_ms) {
  namespace fs = std::filesystem;
  const fs::path summary(summary_path);
  const std::string stem = summary.stem().string();
  const fs::path dir = summary.parent_path().empty() ? fs::path(".") : summary.parent_path();
  fs::create_directories(dir);
  const std::string timeseries_path = (dir / (stem + "_timeseries.csv")).string();
  const std::string insert_path = (dir / (stem + "_insert.csv")).string();
  const std::string events_path = (dir / (stem + "_events.csv")).string();
  const std::string progress_path = (dir / (stem + "_progress.csv")).string();

  std::ofstream insert_csv(insert_path);
  insert_csv << "batch,begin_row,end_row,start_ms,end_ms,insert_latency_ms,insert_throughput_vecps\n";
  for (const auto& record : inserts)
    insert_csv << record.batch << ',' << record.begin_row << ',' << record.end_row << ','
               << record.start_ms << ',' << record.end_ms << ',' << record.latency_ms << ','
               << record.throughput << '\n';

  std::ofstream event_csv(events_path);
  event_csv << "event_id,type,triggering_list,request_ms,start_ms,compute_done_ms,commit_done_ms,end_ms,affected_lists,affected_list_count,split_count,reassigned_vectors\n";
  for (const auto& event : events) {
    std::ostringstream affected;
    for (size_t i = 0; i < event.affected_list_ids.size(); ++i) {
      if (i) affected << ';';
      affected << event.affected_list_ids[i];
    }
    event_csv << event.job_id << ",lire_maintenance," << event.triggering_list << ','
              << event.request_ms << ',' << event.start_ms << ',' << event.compute_done_ms
              << ',' << event.commit_done_ms << ',' << event.end_ms << ','
              << '"' << affected.str() << "\"," << event.affected_lists << ','
              << event.split_count << ','
              << event.reassigned_vectors << '\n';
  }

  std::map<uint32_t, std::vector<const QueryRecord*>> by_frontier;
  for (const auto& record : queries) by_frontier[record.frontier].push_back(&record);
  std::ofstream progress_csv(progress_path);
  progress_csv << "dynamic_insert_progress_pct,frontier_rows,query_count,qps,mean_latency_ms,p50_latency_ms,p95_latency_ms,p99_latency_ms,max_latency_ms,recall_at_k\n";
  std::ofstream timeseries_csv(timeseries_path);
  timeseries_csv << "time_ms,dynamic_insert_progress_pct,frontier_rows,query_count,qps,mean_latency_ms,p50_latency_ms,p95_latency_ms,p99_latency_ms,max_latency_ms,recall_at_k\n";

  std::map<uint64_t, std::vector<const QueryRecord*>> by_time;
  for (const auto& record : queries)
    by_time[static_cast<uint64_t>(record.completion_ms / 200.0)].push_back(&record);
  for (uint32_t frontier : expected_frontiers) {
    const auto found = by_frontier.find(frontier);
    const std::vector<const QueryRecord*> empty;
    const auto& group = found == by_frontier.end() ? empty : found->second;
    // Progress CSV field order is stable and independent of the time-series schema.
    std::vector<double> latencies;
    double recall = 0.0;
    for (const QueryRecord* record : group) {
      latencies.push_back(record->latency_ms);
      recall += record->recall;
    }
    const double mean = latencies.empty() ? 0.0 :
        std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double window_begin = 0.0;
    double window_end = total_ms;
    for (const auto& insert : inserts) {
      if (insert.end_row <= frontier) window_begin = insert.end_ms;
      else { window_end = insert.end_ms; break; }
    }
    const double qps = latencies.size() /
        std::max(0.001, (window_end - window_begin) / 1000.0);
    progress_csv << ProgressForFrontier(frontier, initial_rows, final_rows, first_batch_end)
                 << ',' << frontier << ',' << latencies.size() << ',' << qps << ',' << mean
                 << ',' << Percentile(latencies, .50) << ',' << Percentile(latencies, .95)
                 << ',' << Percentile(latencies, .99) << ','
                 << (latencies.empty() ? 0.0 : *std::max_element(latencies.begin(), latencies.end()))
                 << ',' << (group.empty() ? 0.0 : recall / group.size()) << '\n';
  }
  for (const auto& [bucket, group] : by_time) {
    uint32_t frontier = initial_rows;
    for (const QueryRecord* record : group) frontier = std::max(frontier, record->frontier);
    timeseries_csv << (bucket + 1) * 200 << ',';
    std::vector<double> latencies;
    double recall = 0.0;
    for (const QueryRecord* record : group) {
      latencies.push_back(record->latency_ms);
      recall += record->recall;
    }
    const double mean = latencies.empty() ? 0.0 :
        std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    const double qps = latencies.size() / 0.2;
    timeseries_csv << ProgressForFrontier(frontier, initial_rows, final_rows, first_batch_end)
                   << ',' << frontier << ',' << latencies.size() << ',' << qps << ',' << mean
                   << ',' << Percentile(latencies, .50) << ',' << Percentile(latencies, .95)
                   << ',' << Percentile(latencies, .99) << ','
                   << (latencies.empty() ? 0.0 : *std::max_element(latencies.begin(), latencies.end()))
                   << ',' << (group.empty() ? 0.0 : recall / group.size()) << '\n';
  }

  std::vector<double> all_latencies;
  double recall_sum = 0.0;
  for (const auto& record : queries) {
    all_latencies.push_back(record.latency_ms);
    recall_sum += record.recall;
  }
  const double mean_latency = all_latencies.empty() ? 0.0 :
      std::accumulate(all_latencies.begin(), all_latencies.end(), 0.0) / all_latencies.size();
  const double qps = total_ms > 0.0 ? queries.size() / (total_ms / 1000.0) : 0.0;
  std::ofstream json(summary_path);
  json << "{\n  \"algorithm\": \"Faiss-based LIRE concurrent\",\n"
       << "  \"config\": \"" << JsonEscape(config_path) << "\",\n"
       << "  \"base\": \"" << JsonEscape(base_path) << "\",\n"
       << "  \"query\": \"" << JsonEscape(query_path) << "\",\n"
       << "  \"gt_artifact\": \"" << JsonEscape(gt_path) << "\",\n"
       << "  \"topk\": " << topk << ",\n"
       << "  \"query_workers\": " << query_workers << ",\n"
       << "  \"insert_workers\": " << insert_workers << ",\n"
       << "  \"maintenance_workers\": " << maintenance_workers << ",\n"
       << "  \"query_count\": " << query_count
       << ",\n  \"initial_rows\": " << initial_rows << ",\n  \"final_rows\": " << final_rows
       << ",\n  \"dynamic_insert_vectors\": " << (final_rows - initial_rows)
       << ",\n  \"truncated\": " << (insert_cap_enabled ? "true" : "false")
       << ",\n  \"max_insert_vectors\": ";
  if (insert_cap_enabled) json << max_insert_vectors;
  else json << "null";
  json << ",\n  \"query_records\": " << queries.size() << ",\n  \"query_qps\": " << qps
       << ",\n  \"mean_latency_ms\": " << mean_latency
       << ",\n  \"p50_latency_ms\": " << Percentile(all_latencies, .50)
       << ",\n  \"p95_latency_ms\": " << Percentile(all_latencies, .95)
       << ",\n  \"p99_latency_ms\": " << Percentile(all_latencies, .99)
       << ",\n  \"max_latency_ms\": "
       << (all_latencies.empty() ? 0.0 : *std::max_element(all_latencies.begin(), all_latencies.end()))
       << ",\n  \"recall_at_k\": " << (queries.empty() ? 0.0 : recall_sum / queries.size())
       << ",\n  \"total_time_ms\": " << total_ms << ",\n  \"active_lists\": "
       << index.active_nlist() << ",\n  \"split_count\": " << index.split_count()
       << ",\n  \"reassign_scanned\": " << index.reassign_scanned()
       << ",\n  \"reassign_moved\": " << index.reassign_moved()
       << ",\n  \"outputs\": {\"timeseries\": \"" << JsonEscape(timeseries_path)
       << "\", \"insert\": \"" << JsonEscape(insert_path) << "\", \"events\": \""
       << JsonEscape(events_path) << "\", \"progress\": \"" << JsonEscape(progress_path)
       << "\"}\n}\n";
}

}  // namespace
}  // namespace baseline_eval

int main(int argc, char** argv) {
  using namespace baseline_eval;
  try {
    std::vector<std::string> positional;
    uint64_t max_insert_vectors = std::numeric_limits<uint64_t>::max();
    uint32_t query_threads = 0, insert_threads = 0, maintenance_threads = 0;
    bool cap_seen = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto parse_option = [&](uint32_t* out, const std::string& name) {
        if (i + 1 >= argc) throw std::runtime_error(name + " requires a value");
        const uint64_t value = ParseUint64(argv[++i], name);
        if (value == 0 || value > std::numeric_limits<uint32_t>::max())
          throw std::runtime_error(name + " must be in [1, UINT32_MAX]");
        *out = static_cast<uint32_t>(value);
      };
      if (arg == "--max-insert-vectors") {
        if (cap_seen || i + 1 >= argc) throw std::runtime_error("--max-insert-vectors requires one value");
        max_insert_vectors = ParseUint64(argv[++i], arg);
        cap_seen = true;
      } else if (arg == "--query-threads") parse_option(&query_threads, arg);
      else if (arg == "--insert-threads") parse_option(&insert_threads, arg);
      else if (arg == "--maintenance-threads") parse_option(&maintenance_threads, arg);
      else if (arg.rfind("--", 0) == 0) throw std::runtime_error("Unknown option: " + arg);
      else positional.push_back(arg);
    }
    if (positional.size() < 3 || positional.size() > 4) {
      std::cerr << "Usage: faiss_lire_concurrent_eval <config.json> <dataset_dir_or_base.fvecs> <disp_gt> [summary.json] [--max-insert-vectors N] [--query-threads N] [--insert-threads N] [--maintenance-threads N]\n";
      return 2;
    }
    const std::string config_path = positional[0];
    Config cfg = LoadConfigJson(config_path);
    const std::string base_path = ResolveFileWithSuffix(positional[1], "_base.fvecs");
    const std::string query_path = ResolveFileWithSuffix(
        std::filesystem::path(base_path).parent_path().string(), "_query.fvecs");
    const std::string gt_path = positional[2];
    const std::string summary_path = positional.size() == 4 ? positional[3] :
        (std::filesystem::path("result") / (std::filesystem::path(config_path).stem().string() + "_concurrent.json")).string();
    if (cfg.metric != "l2" && cfg.metric != "L2")
      throw std::runtime_error("The first concurrent LIRE evaluator supports L2 only");
    FvecsData base = LoadFvecs(base_path);
    FvecsData query = LoadFvecs(query_path);
    LimitQueries(&query, cfg.max_queries);
    if (base.d != query.d || (cfg.dim && cfg.dim != base.d))
      throw std::runtime_error("base/query/config dimension mismatch");
    if (query.n == 0 || cfg.topk == 0 || cfg.nprobe == 0 || cfg.nlist == 0)
      throw std::runtime_error("query count, topk, nprobe and nlist must be positive");
    if (cfg.main_index_rows == 0 || cfg.main_index_rows > base.n)
      throw std::runtime_error("main_index_rows must be within the base dataset");
    if (cfg.snapshot_interval == 0) throw std::runtime_error("snapshot_interval must be > 0");
    const uint32_t initial_rows = cfg.main_index_rows;
    const uint64_t remaining = base.n - initial_rows;
    const uint32_t final_rows = static_cast<uint32_t>(initial_rows +
        std::min<uint64_t>(remaining, max_insert_vectors));
    std::vector<uint32_t> frontiers{initial_rows};
    for (uint64_t end = initial_rows; end < final_rows;)
      frontiers.push_back(static_cast<uint32_t>(end = std::min<uint64_t>(final_rows, end + cfg.snapshot_interval)));
    const GroundTruth gt = LoadGroundTruth(gt_path, base_path, query_path, query.n,
        cfg.topk, initial_rows, cfg.snapshot_interval, frontiers);

    LireConfig lire;
    lire.initial_nlist = cfg.nlist;
    lire.max_nlist = cfg.lire_max_nlist ? cfg.lire_max_nlist : cfg.nlist;
    lire.split_threshold = cfg.lire_split_threshold ? cfg.lire_split_threshold :
        std::max<uint32_t>(2, static_cast<uint32_t>((initial_rows + cfg.nlist - 1) / cfg.nlist) * 2);
    lire.reassign_k = cfg.lire_reassign_k;
    lire.seed = cfg.seed;
    lire.max_splits_per_batch = cfg.lire_max_splits_per_batch;
    lire.maintenance_threads = maintenance_threads ? maintenance_threads : 1;
    lire.insert_threads = insert_threads ? insert_threads : 1;
    lire.async_maintenance = true;
    lire.enable_reassign = cfg.lire_enable_reassign;
    lire.validate_invariants = cfg.lire_validate_invariants;
    IndexIVFFlatLIRE index(static_cast<int>(base.d), lire);
    index.nprobe = std::min<faiss::idx_t>(cfg.nprobe, lire.initial_nlist);
    const uint32_t train_rows = cfg.train_rows ? std::min(cfg.train_rows, initial_rows) : initial_rows;
    if (train_rows < cfg.nlist)
      throw std::runtime_error("train_rows must be at least nlist for IVF training");
    index.train(train_rows, RowPtr(base, 0));
    std::vector<faiss::idx_t> initial_ids(initial_rows);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    index.add_with_ids(initial_rows, RowPtr(base, 0), initial_ids.data());
    index.start_maintenance_workers();
    index.wait_for_maintenance();
    index.reset_maintenance_timeline();

    const uint32_t qthreads = query_threads ? query_threads : 1;
    std::atomic<bool> stop_queries{false};
    std::atomic<uint32_t> next_query{0};
    std::mutex records_mutex;
    std::mutex error_mutex;
    std::exception_ptr query_error;
    std::vector<QueryRecord> query_records;
    std::vector<std::thread> workers;
    QueryWorkerJoiner joiner{stop_queries, workers};
    const auto run_start = std::chrono::steady_clock::now();
    auto elapsed_ms = [&] {
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - run_start).count();
    };
    for (uint32_t worker = 0; worker < qthreads; ++worker) {
      workers.emplace_back([&, worker] {
        (void)worker;
        try {
          std::vector<float> distances(cfg.topk);
          std::vector<faiss::idx_t> labels(cfg.topk);
          while (!stop_queries.load(std::memory_order_relaxed)) {
            const uint32_t qi = next_query.fetch_add(1, std::memory_order_relaxed) % query.n;
            const uint32_t frontier = static_cast<uint32_t>(index.published_rows());
            const auto start = std::chrono::steady_clock::now();
            index.search_at_frontier(1, RowPtr(query, qi), cfg.topk,
                                     distances.data(), labels.data(), frontier);
            const auto end = std::chrono::steady_clock::now();
            QueryRecord record;
            record.query_id = qi;
            record.frontier = frontier;
            record.latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
            record.completion_ms = elapsed_ms();
            const auto it = std::lower_bound(gt.frontiers.begin(), gt.frontiers.end(), frontier);
            if (it == gt.frontiers.end() || *it != frontier)
              throw std::runtime_error("query observed a frontier absent from GT artifact");
            const size_t version = static_cast<size_t>(it - gt.frontiers.begin());
            record.gt_version = static_cast<uint32_t>(version);
            record.returned_topk = labels;
            std::unordered_set<uint32_t> expected(gt.ids[version][qi].begin(), gt.ids[version][qi].end());
            uint32_t hits = 0;
            for (faiss::idx_t id : labels)
              if (id >= 0 && expected.count(static_cast<uint32_t>(id))) ++hits;
            record.recall = cfg.topk ? static_cast<double>(hits) / cfg.topk : 0.0;
            std::lock_guard<std::mutex> lock(records_mutex);
            query_records.push_back(record);
          }
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!query_error) query_error = std::current_exception();
          stop_queries.store(true, std::memory_order_relaxed);
        }
      });
    }

    std::vector<InsertRecord> insert_records;
    const uint32_t first_batch_end = final_rows > initial_rows ?
        std::min<uint32_t>(final_rows, initial_rows + cfg.snapshot_interval) : initial_rows;
    uint32_t batch_id = 0;
    for (uint32_t begin = initial_rows; begin < final_rows;) {
      const uint32_t end = static_cast<uint32_t>(std::min<uint64_t>(final_rows,
          static_cast<uint64_t>(begin) + cfg.snapshot_interval));
      InsertRecord record;
      record.batch = batch_id++;
      record.begin_row = begin;
      record.end_row = end;
      record.start_ms = elapsed_ms();
      const auto insert_start = std::chrono::steady_clock::now();
      std::vector<faiss::idx_t> ids(end - begin);
      std::iota(ids.begin(), ids.end(), begin);
      index.add_with_ids(end - begin, RowPtr(base, begin), ids.data());
      const auto insert_end = std::chrono::steady_clock::now();
      record.end_ms = elapsed_ms();
      record.latency_ms = std::chrono::duration<double, std::milli>(insert_end - insert_start).count();
      record.throughput = record.latency_ms > 0.0 ?
          (end - begin) * 1000.0 / record.latency_ms : 0.0;
      insert_records.push_back(record);
      begin = end;
    }
    try {
      index.wait_for_maintenance();
    } catch (...) {
      stop_queries.store(true, std::memory_order_relaxed);
      for (auto& worker : workers) if (worker.joinable()) worker.join();
      throw;
    }
    stop_queries.store(true, std::memory_order_relaxed);
    for (auto& worker : workers) if (worker.joinable()) worker.join();
    const double total_ms = elapsed_ms();
    {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (query_error) std::rethrow_exception(query_error);
    }
    index.stop_maintenance_workers();
    WriteOutputs(summary_path, config_path, base_path, query_path, gt_path,
        cfg.topk, qthreads, lire.insert_threads, lire.maintenance_threads,
        initial_rows, final_rows, first_batch_end, frontiers, cap_seen,
        max_insert_vectors, query.n,
        query_records, insert_records, index.maintenance_events(), index, total_ms);
    std::cout << "Concurrent LIRE run completed: rows=" << final_rows
              << " queries=" << query_records.size() << " summary=" << summary_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
