# Concurrent Faiss-based LIRE evaluator

The concurrent entry point is `build/faiss_lire_concurrent_eval`. It reads the
existing LIRE JSON config, the dataset directory (with `_base.fvecs` and
`_query.fvecs`), and a DiSPQ concurrent GT artifact:

```bash
./build/faiss_lire_concurrent_eval CONFIG.json DATASET_DIR DISP_GT.gt SUMMARY.json \
  --max-insert-vectors 30000 \
  --query-threads 4 --insert-threads 1 --maintenance-threads 2
```

`--max-insert-vectors` limits only the rows after `main_index_rows`; the capped
run constructs and validates its own exact frontier sequence against the GT
artifact. `snapshot_interval` is the dynamic insert batch size. The evaluator
rejects a GT artifact whose identities, query count, top-k, initial row count,
batch size, frontier count, or individual frontiers differ.

The first implementation supports raw L2 fvecs. Prepare the shared GT with the
same base/query paths, query count, top-k, initial row count, batch size, and
frontiers, and with preprocessing disabled. Query workers scan with one
software thread each; `insert_threads` controls Faiss/OpenMP coarse assignment;
`maintenance_threads` controls background LIRE jobs. Choose those values so
their sum fits the experiment's CPU budget.

The evaluator writes `SUMMARY.json`, `SUMMARY_timeseries.csv`,
`SUMMARY_insert.csv`, `SUMMARY_events.csv`, and `SUMMARY_progress.csv`. The
CSV data is available in this first version; plot generation is not included.
