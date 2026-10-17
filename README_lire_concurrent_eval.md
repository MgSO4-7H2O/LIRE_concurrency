# Concurrent Faiss-based LIRE evaluator

`build/faiss_lire_concurrent_eval` runs a concurrent Faiss-based reproduction
of LIRE. Initial build and its maintenance drain finish before measurement.
During measurement, one sequential foreground insert stream, fixed query
workers, and background LIRE maintenance run together. After the last dynamic
insert batch completes, the evaluator stops admitting queries and joins all
in-flight queries. It records the streaming workload end at that point, then
drains residual maintenance separately. That residual drain is excluded from
query count, QPS, latency, recall, progress, and streaming workload duration.

The configuration selects one of three offline/runtime modes. GT and initial
index paths, query/insert/maintenance thread allocation, update thread budget,
maintenance backlog, insert pacing, and summary output path are all read from
JSON, including `concurrent_dataset_path`. Runtime options are no longer
passed on the command line. For SIFT1M:

```bash
mkdir -p artifacts result
./build-concurrent/faiss_lire_concurrent_eval \
  configs/sift1m_concurrent_gt_prepare.json
```

Prepare and persist the fully built initial index (`main_index_rows`) separately:

```bash
./build-concurrent/faiss_lire_concurrent_eval \
  configs/sift1m_concurrent_init_prepare.json
```

The query config loads both artifacts and starts at the initial frontier without
training or inserting those initial rows again:

```bash
./build-concurrent/faiss_lire_concurrent_eval \
  configs/sift1m_concurrent_query.json
```

Set `concurrent_gt_mode` to `prepare`, `load`, or `off`, and
`concurrent_init_mode` to `prepare`, `load`, or `off`. Query runs use `load`
for both. An init preparation uses GT `off` and init `prepare`; GT preparation
uses GT `prepare` and init `off`. `concurrent_gt_path` and
`concurrent_init_path` specify the artifacts. The initial index artifact stores
the complete LIRE list/topology state and validates the base prefix hash,
dimension, list capacities, and row count when loaded.

GT prepare mode computes exact squared-L2 over the original float32 rows,
`sum((q[i] - x[i])^2)`, with no whitening, normalization, PQ, or IVF search.
For each query it scans only rows added since the previous frontier and
maintains an exact top-k heap, snapshotting the ordered IDs at each frontier.
OpenMP parallelizes independent queries during this offline phase. The
`LIREGT01` artifact records the raw-L2 distance-space identifier, canonical
base/query paths and content fingerprints, query count, dimension, top-k,
initial rows, snapshot interval, configured final rows, the complete frontier
sequence, per-frontier DocIds, and a payload checksum. The query evaluator only
loads `LIREGT01`; it rejects DiSPQ GT files, mismatched metadata/data/frontiers,
corrupt payloads, and trailing bytes without fallback.

The GT prepare and query configs share `topk`, `main_index_rows`,
`snapshot_interval`, and `final_rows`. Queries capture `published_rows()` once
and compare their results only with the exact GT for that same frontier.
Maintenance changes list organization but never advances the GT frontier.

Thread allocation uses `concurrent_query_workers`,
`concurrent_query_threads_per_request`, `concurrent_insert_threads`, and
`concurrent_maintenance_threads`. Set `concurrent_update_thread_budget` to at
least the sum of the insert and maintenance threads. The external pacing cap
uses `concurrent_insert_rate_limit_vecps`; `concurrent_max_insert_vectors` can
limit a run to a prepared GT frontier.

Build command:

```bash
cmake --build build-concurrent --target faiss_lire_concurrent_eval -j
```

Command shape:

```bash
./build-concurrent/faiss_lire_concurrent_eval CONFIG.json
```

Align `concurrent_query_workers` and
`concurrent_query_threads_per_request` with the DiSPQ configuration, as well
as the external insert rate cap. `concurrent_query_workers N`
means N concurrent logical query requests. Each request uses a fixed search
team of `concurrent_query_threads_per_request M` threads, so its Query-side
budget is N × M. Query resources do not change when the update-side split
changes.

Each query team is created before the streaming workload and reused for every
request handled by its outer query worker. The calling worker is team member
0 and M−1 persistent helper threads scan the request's selected IVF lists.
Coarse routing and the visible frontier are selected once per request; all
team members scan disjoint lists from that route set, and their local top-k
results are merged into the same top-k ranking as the existing single-thread
LIRE path. This custom LIRE search path
does not call Faiss's native IVF search: `search_at_frontier(1, ...)` was a
serial list scan, so the evaluator parallelizes the already selected list
scan explicitly. No OpenMP thread-count setting is used in Query. With M=1,
the current serial scan path is retained. The logged `query_thread_budget` is
`query_workers × query_threads_per_request`; helper work is not counted as
additional logical queries.

`concurrent_insert_threads` configures the Faiss/OpenMP
coarse assignment inside the single foreground insert stream; it does not
create N frontend insert workers. `concurrent_maintenance_threads` starts N
LIRE background workers. `concurrent_update_thread_budget` checks that
`insert_threads + maintenance_threads` does not exceed the update-side CPU
budget. Query workers are accounted for separately. Faiss/OpenMP nested
parallelism is disabled for the insert assignment region; maintenance jobs do
not create nested OpenMP teams.

The external insert rate is a batch-arrival cap in vectors per second. For
batch size B and rate R, successive batch release times are spaced by B/R
seconds, matching DiSPQ's `concurrent_target_insert_vecps` pacing. A slow batch
does not get skipped. If backlog backpressure delays admission, the next
release schedule restarts from that admission point, as it does in DiSPQ.
Measured `insert_throughput_vecps` remains the actual time spent in
`add_with_ids()` and is not the configured arrival rate.

`concurrent_max_maintenance_backlog` bounds the combined pending SplitJob and
ReassignJob queue before admitting the next dynamic batch. When the queue
exceeds N, the foreground stream waits until it falls to N or fewer; Query and
maintenance continue during that wait. Split jobs for the same physical list
are deduplicated. One split can enqueue many per-vector reassignments, so the
queue can briefly exceed the threshold by that split's fan-out; the threshold
gates subsequent batch admission. A value of zero selects twice the number of
maintenance workers (minimum one).

The internal maintenance pipeline follows SPFresh/LIRE's split and
reassignment shape. A SplitJob locks its source posting and a reserved child
posting in ascending list-ID order, reads the current source contents, filters
copies whose logical membership points elsewhere, and runs the existing
balanced K=2 split. Disjoint postings can split concurrently. Only short
topology publication takes the global state lock. Lock order is posting locks
in ascending ID order, then the state lock, then membership. Inserts and
per-vector reassignment use the same order, avoiding a state/list ABBA cycle.

After publication, candidate discovery scans the two children and up to
`lire_reassign_k` nearby lists. It only submits individual ReassignJobs; its
snapshot is not a commit transaction. ReassignJob carries the candidate vector
payload captured during CollectReAssign, following the behavior observed in
SPFresh's ReassignAsync path. Vector contents are immutable in this insert-only
evaluator, so a job validates logical membership and routes the captured
payload against current active centroids without rescanning the old posting to
recover the vector. It locks the previous and target postings in ascending
order, appends the physical target copy if needed, and then updates the
authoritative `membership[id]`. A job whose previous membership has changed is
dropped cheaply. Existing obsolete copies can remain in old postings; Query
filters a copy unless `membership[id]` equals the scanned list, and later
splits reclaim such copies. This single-assignment Faiss adaptation does not
copy SPFresh's replicated SPANN layout or VersionLabel representation: logical
membership is the version/location authority, with one current posting per
vector.

This evaluator uses one configured maintenance pool for both SplitJobs and
ReassignJobs. In the inspected SPFresh source, `ReassignAsync()` submits to
`m_splitThreadPool`; although a separate reassign pool is initialized and
included in `AllFinished()`, that method is not its submission path. The local
single-pool queue therefore follows the observed submission path without
adding unbudgeted workers. `topology_generation_` is used only to refresh a
vector's routing decision if topology changes while it waits for posting
locks; it does not invalidate or restart neighborhood discovery. The summary
and insert CSV expose split/reassign job counts, stale job drops, successful
moves, `query_stale_copies_skipped`, and
`maintenance_stale_copies_filtered`. These are filter occurrences, not unique
physical-copy counts. The compatibility field `stale_physical_copies_filtered`
is their sum. `reassign_source_rescans` is expected to remain zero, while
`reassign_payload_bytes_enqueued` reports the cumulative vector payload bytes
stored in accepted ReassignJobs. Query routing validates its
topology snapshot after acquiring the selected posting locks; it reroutes if a
split published in the gap, so it never scans an old route set against newly
published split postings.

The evaluator writes `SUMMARY.json`, `SUMMARY_timeseries.csv`,
`SUMMARY_insert.csv`, `SUMMARY_events.csv`, and `SUMMARY_progress.csv`.
Progress contains one point per actual dynamic insert batch. The initial-only
GT frontier remains available to queries but is not a progress point. The
first completed dynamic batch is 0%; the last is 100% (a sole batch is 100%).
Each point groups query completions in that batch's `[start_ms, end_ms]`
interval and divides its query count by the batch duration. Empty intervals
write zero query count/QPS and blank recall/latency. Streaming QPS uses the
workload duration through query worker join, never the later maintenance
drain.

The summary records streaming start/end/duration, maintenance drain start/end
and duration, split/reassign job counts, stale reassign drops, successful
moves, stale physical copies filtered, peak and stream-end backlog, queue wait,
and execution time. The insert CSV carries those counters and backlog
snapshots. The events CSV retains one event per SplitJob; aggregate queue-wait
and execution totals include both SplitJobs and ReassignJobs.
`compute_done_ms`, `commit_done_ms`, and `end_ms` remain coarse job timestamps
and do not claim independent phase measurements.

For fair comparison, keep query workers, query threads per request, dataset,
GT, batch schedule, external insert rate, and total update-side thread budget
aligned with DiSPQ. The insert/maintenance thread split can be tuned for LIRE's
foreground-updater/background-local-rebuilder pipeline; DiSPQ's internal
maintenance model is not imported. The split, neighbor discovery, reassignment
conditions, and local publication remain LIRE-specific. This evaluator supports
raw L2 fvecs and insert-only workloads; deletion and plotting are not
implemented. AVX and formal performance tuning remain deferred.

For small functional checks, set `concurrent_max_insert_vectors`,
`concurrent_query_workers` to `2` or `4`,
`concurrent_query_threads_per_request` to `1` or `3`, and choose insert and
maintenance thread counts that fit the update budget. Enable
`lire_validate_invariants` in a small run to validate
membership and inverted-list consistency after initial build and final
maintenance drain. The supplied SIFT1M candidate config uses 4,096 lists,
210,000 initial rows, 10,000-row batches, `lire_split_threshold=384`, and
`lire_reassign_k=64`. This threshold is held fixed for the execution-model
check, not claimed as a final tuned value. Its
`lire_validate_invariants=true` setting checks membership/list consistency
after the initial build and after the streaming maintenance drain.
