# Concurrent Faiss-based LIRE evaluator

`build/faiss_lire_concurrent_eval` runs a concurrent Faiss-based reproduction
of LIRE. Initial build and its maintenance drain finish before measurement.
During measurement, one sequential foreground insert stream, fixed query
workers, and background LIRE maintenance run together. After the last dynamic
insert batch completes, the evaluator stops admitting queries and joins all
in-flight queries. It records the streaming workload end at that point, then
drains residual maintenance separately. That residual drain is excluded from
query count, QPS, latency, recall, progress, and streaming workload duration.

Example:

```bash
./build/faiss_lire_concurrent_eval CONFIG.json DATASET_DIR DISP_GT.gt SUMMARY.json \
  --max-insert-vectors 30000 \
  --query-workers 8 \
  --query-threads-per-request 3 \
  --insert-threads 1 \
  --maintenance-threads 3 \
  --update-thread-budget 4 \
  --insert-rate-limit 10000 \
  --max-maintenance-backlog 6
```

Align `--query-workers` and `--query-threads-per-request` with the DiSPQ
configuration, as well as the external insert rate cap. `--query-workers N`
means N concurrent logical query requests. Each request uses a fixed search
team of `--query-threads-per-request M` threads, so its Query-side budget is
N × M. The deprecated `--query-threads N` remains an alias for
`--query-workers N`; it does not configure per-request parallelism. Query
resources do not change when the update-side split changes.

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

`--insert-threads N` configures the Faiss/OpenMP
coarse assignment inside the single foreground insert stream; it does not
create N frontend insert workers. `--maintenance-threads N` starts N LIRE
background workers. `--update-thread-budget` checks that
`insert_threads + maintenance_threads` does not exceed the update-side CPU
budget. Query workers are accounted for separately. Faiss/OpenMP nested
parallelism is disabled for the insert assignment region; maintenance jobs do
not create nested OpenMP teams. If omitted, the update-side budget defaults to
2, matching the default one insert thread plus one maintenance thread; larger
thread splits require an explicit budget.

The external insert rate is a batch-arrival cap in vectors per second. For
batch size B and rate R, successive batch release times are spaced by B/R
seconds, matching DiSPQ's `concurrent_target_insert_vecps` pacing. A slow batch
does not get skipped. If backlog backpressure delays admission, the next
release schedule restarts from that admission point, as it does in DiSPQ.
Measured `insert_throughput_vecps` remains the actual time spent in
`add_with_ids()` and is not the configured arrival rate.

`--max-maintenance-backlog N` is the queued-job threshold used before admitting
the next dynamic batch. When the queue exceeds N, the foreground stream waits
until it falls to N or fewer; Query and maintenance continue during that wait.
Jobs are deduplicated by physical list. One batch can enqueue work for several
lists, so the queue can briefly exceed the threshold by that batch's fan-out;
the threshold gates subsequent batch admission and prevents debt growing
without bound across batches. If omitted, the threshold is twice the number of
maintenance workers (minimum one).

Maintenance work on separate lists can compute in parallel. A split records
the source-list generation and centroid-topology generation before clustering.
Before publishing children it validates both again. Reassignment snapshots
the relevant child and neighbor list generations plus topology generation;
publication checks them before applying moves. Stale split work is requeued;
stale reassignment computation is discarded and retried from a fresh snapshot.
Jobs for the same physical list are deduplicated. Multi-list locks are always
acquired by ascending list ID. The shared state lock still serializes the
short publication step and protects the Faiss active-centroid index; clustering
and candidate discovery run outside its exclusive section. Reassignment
publication currently materializes and rewrites affected lists while holding
that lock, so query routing can pause during this commit.

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
and duration, maintenance jobs enqueued/started/completed/retried, peak and
stream-end backlog, queue wait, and execution time. The insert CSV carries
backlog snapshots; the events CSV includes per-job queue wait, execution, and
stale retry counts. `compute_done_ms`, `commit_done_ms`, and `end_ms` remain
coarse job timestamps and do not claim independent phase measurements.

For fair comparison, keep query workers, query threads per request, dataset,
GT, batch schedule, external insert rate, and total update-side thread budget
aligned with DiSPQ. The insert/maintenance thread split can be tuned for LIRE's
foreground-updater/background-local-rebuilder pipeline; DiSPQ's internal
maintenance model is not imported. The split, neighbor discovery, reassignment
conditions, and local publication remain LIRE-specific. This evaluator supports
raw L2 fvecs and insert-only workloads; deletion and plotting are not
implemented. AVX and formal performance tuning remain deferred.

For small functional checks, use `--max-insert-vectors`, `--query-workers 2`
or `4`, `--query-threads-per-request 1` or `3`, `--insert-threads 1`, and a maintenance-thread count that fits the
update budget. Enable `lire_validate_invariants` in a small run to validate
membership and inverted-list consistency after initial build and final
maintenance drain. The supplied SIFT1M config's current 4,096-list,
210,000-initial-row, 10,000-row batch, and 128 split threshold values are
unchanged here; collect backlog and drain measurements before tuning the
trigger threshold.
