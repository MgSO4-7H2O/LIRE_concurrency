# CPU IndexIVFFlat + LIRE baseline

This target keeps the existing `run_eval` control flow and metrics, but replaces
static IVF cluster membership with synchronous LIRE split/reassign maintenance.
It supports only L2, sequential batch insertion, and search snapshots. It does
not implement deletion, SPDK, replicas, version maps, or asynchronous rebuilds.

## Source files

- `src/faiss_ivfflat_lire.h`
- `src/faiss_ivfflat_lire.cpp`
- `src/faiss_ivfflat_lire_run_eval.cpp`
- `src/faiss_baseline_run_eval_common.h` (factory and config integration)
- `configs/ivfflat_lire_example.json`

## Build

The project expects the Faiss source tree at `third_party/faiss`, exactly like
the original baseline bundle.

```bash
cd faiss_baseline
# When the Faiss submodule/source tree is not present:
git clone https://github.com/facebookresearch/faiss.git third_party/faiss

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target faiss_ivfflat_lire_run_eval -j
```

The target is CPU-only because the top-level CMake configuration sets
`FAISS_ENABLE_GPU=OFF` and links the CPU Faiss target plus OpenMP.

## Run

```bash
./build/faiss_ivfflat_lire_run_eval \
  configs/ivfflat_lire_example.json \
  /path/to/dataset_dir
```

The dataset directory must contain files ending in `_base.fvecs` and
`_query.fvecs`. An explicit output path is optional:

```bash
./build/faiss_ivfflat_lire_run_eval \
  configs/ivfflat_lire_example.json \
  /path/to/dataset_dir \
  result/lire_online_eval.json
```

The output keeps the original IVFFlat fields and meanings: recall, recall_new,
recall_old, query QPS, average query latency, avg_scanned, update_ms, and update
throughput. `update_ms` includes the synchronous split/reassign work, so each
search snapshot sees a fully committed index state.

## LIRE parameters

- `nlist`: initial number of trained IVF centroids.
- `nprobe`: number of active lists probed per query.
- `lire_max_nlist`: fixed physical list/centroid capacity. Every split consumes
  one spare slot.
- `lire_split_threshold`: split a list when its size is greater than this value.
- `lire_reassign_k`: scan this many old-centroid neighboring lists after split.
- `lire_max_splits_per_batch`: hard safety bound for cascading splits.
- `lire_enable_reassign`: false gives a split-only ablation.
- `lire_validate_invariants`: performs an O(N) post-batch check that every ID in
  `[0, ntotal)` appears exactly once and its membership is correct.

Choose `lire_max_nlist` with enough headroom for the full stream. If it is
exhausted, or the cascade safety limit is reached, the evaluator stops instead
of emitting measurements from a partially maintained index.

## Implementation correspondence with SPFresh

The implementation intentionally copies only the cluster-maintenance logic.

| LIRE operation | This baseline | SPFresh source counterpart |
|---|---|---|
| Initial clustering | temporary standard Faiss `IndexIVFFlat::train` | SPANN initial balanced index construction |
| Oversize trigger | `add_with_ids` -> `run_maintenance` | `ExtraDynamicSearcher::Append` -> `SplitAsync` |
| Local split | `split_one` | `ExtraDynamicSearcher::Split` |
| K=2 balanced clustering | `split_balanced_kmeans` | `KmeansArgs(2)` + `KmeansClustering(..., 1000, 100.0F, ...)` |
| Split-list condition | `collect_and_reassign`, child branch | `CheckIsNeedReassign(..., isInSplitHead=true)` |
| Neighbor-list condition | `collect_and_reassign`, neighbor branch | `CheckIsNeedReassign(..., isInSplitHead=false)` |
| Exact reassignment | `nearest_active` + `apply_moves` | `Reassign` after candidate filtering |
| Cascading split | `run_maintenance` queue | split caused by reassignment append |

Differences are limited to the requested baseline scope:

1. SPFresh keeps multiple SPANN replicas selected by RNG; this baseline uses
   one standard IVF assignment, the globally nearest active L2 centroid.
2. SPFresh appends a new version and lazily garbage-collects the old replica;
   this in-memory baseline physically moves the record, so the evaluator always
   sees exactly one copy of each ID.
3. SPFresh executes maintenance in background jobs; this baseline executes it
   synchronously under one write lock. This preserves the existing `run_eval`
   batch-insert/search ordering and prevents GT/recall evaluation from observing
   an intermediate split state.
4. SPFresh posting heads can be data vectors. Faiss IVF uses virtual k-means
   centroids, so the local split retains the computed L2 means.
