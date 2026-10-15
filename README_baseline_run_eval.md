# Faiss baseline run_eval scripts

This bundle contains Faiss baseline evaluators aligned to the DISPQ `run_eval` and
`run_eval_ms` interfaces.

## Files

- `src/faiss_ivfflat_run_eval.cpp`
- `src/faiss_ivfflat_run_eval_ms.cpp`
- `src/faiss_ivfpq_run_eval.cpp`
- `src/faiss_ivfpq_run_eval_ms.cpp`
- `src/faiss_ivfsq_run_eval.cpp`
- `src/faiss_ivfsq_run_eval_ms.cpp`
- `src/faiss_ivfrabitq_run_eval.cpp`
- `src/faiss_ivfrabitq_run_eval_ms.cpp`
- `src/faiss_opq_run_eval.cpp`
- `src/faiss_opq_run_eval_ms.cpp`
- `src/faiss_baseline_run_eval_common.h`
- `CMakeLists.txt`

`CMakeLists.txt` keeps the original Faiss/OpenMP linking pattern and only changes
the target/source names.

## Run commands

Normal streaming evaluator:

```bash
./build/faiss_ivfflat_run_eval configs/ivfflat_sift.json ../data/sift/origin
./build/faiss_ivfpq_run_eval configs/ivfpq_sift.json ../data/sift/origin
./build/faiss_ivfsq_run_eval configs/ivfsq_sift.json ../data/sift/origin
./build/faiss_ivfrabitq_run_eval configs/ivfrabitq_sift.json ../data/sift/origin
./build/faiss_opq_run_eval configs/opq_sift.json ../data/sift/origin
```

MS workload evaluator:

```bash
./build/faiss_ivfflat_run_eval_ms configs/ivfflat_ms.json ../data/msturing30m/fvecs ../data/msturing30m/workload/final_runbook.json
./build/faiss_ivfpq_run_eval_ms configs/ivfpq_ms.json ../data/msturing30m/fvecs ../data/msturing30m/workload/final_runbook.json
./build/faiss_ivfsq_run_eval_ms configs/ivfsq_ms.json ../data/msturing30m/fvecs ../data/msturing30m/workload/final_runbook.json
./build/faiss_ivfrabitq_run_eval_ms configs/ivfrabitq_ms.json ../data/msturing30m/fvecs ../data/msturing30m/workload/final_runbook.json
./build/faiss_opq_run_eval_ms configs/opq_ms.json ../data/msturing30m/fvecs ../data/msturing30m/workload/final_runbook.json
```

Both forms also support the minimal DISPQ-style invocation:

```bash
./build/faiss_ivfflat_run_eval <config.json> <dataset_dir>
./build/faiss_ivfflat_run_eval_ms <config.json> <dataset_dir>
```

For `_run_eval_ms`, when the workload path is omitted, the program searches for a
file ending with `_workload.json` under `dataset_dir`.

The default result path is:

```text
result/<dataset>/<metric_path>/<baseline_algorithm>/<config_name>/online_eval.json
```

This follows the DISPQ dataset/config path rule and adds the baseline algorithm
level to avoid collisions between IVFFlat, IVFPQ and IVFRaBitQ with the same
config name. A final explicit output path may also be passed on the command line.

## Common config fields

Required in practice:

- `metric`: `l2` or `cosine`. For `cosine`, base/query vectors are L2-normalized
  and searched with normalized L2 ranking, which is equivalent for unit vectors.
- `topk`: recall K.
- `nlist`: IVF coarse clusters.
- `nprobe`: IVF probes per query.
- `train_rows`: number of leading base vectors used for Faiss training. If `0`,
  `_run_eval` uses `main_index_rows`; `_run_eval_ms` uses `main_index_rows` when
  positive.
- `max_queries`: truncate query set; `0` means use all queries.
- `add_threads`, `search_threads`, `gt_threads`: OpenMP thread controls. `0`
  means use the runtime maximum.

Fields used by `_run_eval`:

- `main_index_rows`: initial indexed prefix size.
- `snapshot_interval`: each later insert batch size; every insert batch is
  followed by one search snapshot.

Fields used by `_run_eval_ms`:

- The JSON workload controls all insert/search checkpoints.
- `main_index_rows` is only used as the default training size when `train_rows=0`.

IVFPQ-only fields:

- `M`: number of PQ subquantizers.
- `nbits`: PQ bits per subquantizer.
- `rerank_factor`: retrieve `topk * rerank_factor` PQ candidates before exact rerank.
- `exact_rerank`: `true` by default. If false, recall is raw IVFPQ ADC recall.
- `unsafe_parallel_add`: false by default. IVFPQ online add uses one thread unless this is true.

IVFSQ-only fields:

- `sq_type`: scalar quantizer type. Supported values are `QT_8bit`, `QT_4bit`,
  `QT_6bit`, `QT_fp16`, and `QT_8bit_uniform`.
- `rerank_factor`: default `1`. Use values above 1 to retrieve more IVFSQ
  candidates and exact-rerank them.
- `exact_rerank`: `true` by default, but has no effect when `rerank_factor=1`.

IVFRaBitQ-only fields:

- `nb_bits`: database RaBitQ bits per dimension.
- `qb`: query quantization bits.
- `centered`: passed to `IVFRaBitQSearchParameters`.
- `rerank_factor`: retrieve `topk * rerank_factor` RaBitQ candidates before exact rerank.
- `exact_rerank`: `true` by default.

OPQ+IVFPQ-only fields:

- `M`: number of PQ subquantizers.
- `nbits`: PQ bits per subquantizer.
- `opq_M`: OPQ subquantizers. If `0`, defaults to `M`.
- `opq_niter`: OPQ outer iterations.
- `opq_niter_pq`: PQ iterations per OPQ outer iteration.
- `opq_niter_pq_0`: initial PQ iterations for OPQ.
- `opq_max_train_points`: max OPQ training points.
- `rerank_factor`: retrieve `topk * rerank_factor` OPQ+IVFPQ candidates before exact rerank.
- `exact_rerank`: `true` by default.
- `unsafe_parallel_add`: false by default. Online add uses one thread unless this is true.

## Output metrics

The JSON contains `final_metrics`, `summary`, and `snapshots`.

Baseline records the DISPQ-compatible fields that have the same physical meaning:

- `recall`, `recall@K`
- `recall_new`, `recall_old`, `gt_new_ratio`
- `gt_new_total`, `gt_old_total`, `hit_new_total`, `hit_old_total`
- `qps`, `query_qps`, `latency_ms` or `avg_query_ms`
- `update_ms`, `cumulative_update_ms`, `update_throughput_vecps`
- `avg_scanned`
- `memory_rss_bytes`, `memory_hwm_bytes`

DISPQ-specific merge/global-rebuild fields are intentionally not emitted for
these baselines, because Faiss IVF online add has no corresponding merge or
global rebuild operation in this evaluator.

## IndexIVFFlat + LIRE target

A CPU-only LIRE cluster-maintenance baseline is included as:

```text
src/faiss_ivfflat_lire_run_eval.cpp
src/faiss_ivfflat_lire.h
src/faiss_ivfflat_lire.cpp
```

Build and run it with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target faiss_ivfflat_lire_run_eval -j
./build/faiss_ivfflat_lire_run_eval configs/ivfflat_lire_example.json <dataset_dir>
```

See `README_lire_baseline.md` for the maintenance semantics, configuration, and
mapping to the SPFresh implementation.
