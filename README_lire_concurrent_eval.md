# Concurrent Faiss-based LIRE evaluator

`build/faiss_lire_concurrent_eval` is the first functional concurrent LIRE
version. Query workers, the foreground insertion stream, and the background
LIRE maintenance worker can overlap. Inserts are dispatched as one sequential
batch stream; the evaluator does not run multiple frontend `add_with_ids()`
workers.

The evaluator reads an existing LIRE JSON config, a dataset directory (with
`_base.fvecs` and `_query.fvecs`), and a DiSPQ concurrent GT artifact:

```bash
./build/faiss_lire_concurrent_eval CONFIG.json DATASET_DIR DISP_GT.gt SUMMARY.json \
  --max-insert-vectors 30000 \
  --query-threads 4 --insert-threads 1 --maintenance-threads 1
```

`--max-insert-vectors` limits only rows after `main_index_rows`; the capped run
constructs and validates its own exact frontier sequence against the GT
artifact. `snapshot_interval` is the dynamic insert batch size. GT validation
checks magic/version, base and query path identities, query count, top-k,
initial row count, batch size, frontier count, every frontier, and trailing
bytes. Generate the artifact using the same resolved base/query paths and
workload parameters.

This first version supports raw L2 fvecs. Query workers run Faiss search with
one software thread each. `--insert-threads` sets the insert-side
Faiss/OpenMP coarse-assignment thread count; it does not create frontend insert
workers. There is one foreground insert stream and its batches are dispatched
sequentially. `--maintenance-threads` currently accepts only `1`: concurrent
maintenance workers need complete generation validation for neighboring-list
and centroid snapshots, which is not implemented in this phase. Use
`--query-threads 2` to `4`, `--insert-threads 1`, and
`--maintenance-threads 1` for the first functional run.

`*_progress.csv` contains one row per completed dynamic insert batch. Its
progress is calculated from actual `end_row` values: the first completed
dynamic batch is 0% and the final completed batch is 100%. A run with one
dynamic batch assigns that point 100%; a run with no dynamic inserts has no
progress data rows. The initial-only frontier remains in the GT sequence for
queries before the first batch completes, but it is not a progress point.
Progress metrics group queries by completion time in that batch's
`[start_ms, end_ms]` interval. QPS is `query_count / batch_duration_seconds`.
For an interval with no completed queries, query count and QPS are 0 and the
recall and latency fields are empty.

Maintenance event timestamps currently describe coarse job timing;
`compute_done_ms`, `commit_done_ms`, and `end_ms` do not independently measure
precise compute and publication phases. `apply_moves()` retains a conservative
global state lock while it materializes and rewrites affected lists, so query
routing can pause during reassignment commit. This is retained for the
first correctness pass and can be revisited after SIFT1M measurements.

The evaluator writes `SUMMARY.json`, `SUMMARY_timeseries.csv`,
`SUMMARY_insert.csv`, `SUMMARY_events.csv`, and `SUMMARY_progress.csv`. Plot
generation is not included. Formal performance tuning and AVX target tuning
have not started; the current CMake Faiss CPU target is unchanged. Deletion is
not supported.

## SIFT1M functional configuration

`configs/ivfflat_lire_concurrent_sift1m.json` is sized for a small functional
check: 50,000 initial rows, 5,000-row dynamic batches, and a split threshold
chosen so a 30,000-row insertion cap is likely to trigger maintenance. Prepare
the matching DiSPQ GT artifact with 1,000 queries, top-k 10, raw L2, initial
rows 50,000, batch size 5,000, and frontiers 50,000 through 80,000 in steps of
5,000. Pass `--max-insert-vectors 30000` to use that same truncated frontier
range.
