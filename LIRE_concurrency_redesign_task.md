# LIRE Concurrency Redesign Task

## 1. Purpose

This document defines the implementation task for adapting the local Faiss-based LIRE reproduction in:

- `MgSO4-7H2O/LIRE_concurrency`

to support a concurrent workload in which:

- Query
- Insert
- LIRE maintenance

can overlap during execution.

The redesign must be guided primarily by the official SPFresh implementation and SPFresh/LIRE execution model:

- Official repository: `SPFresh/SPFresh`
- Relevant official components include:
  - `AnnService/inc/Core/SPANN/ExtraDynamicSearcher.h`
  - `AnnService/inc/SPFresh/SPFresh.h`
  - `AnnService/inc/Core/SPANN/Options.h`
  - `AnnService/inc/Core/SPANN/ParameterDefinitionList.h`

The local system is intentionally a **Faiss-based reproduction of LIRE**, not a complete reimplementation of the full SPFresh serving stack.

The implementation should preserve this scope.

---

## 2. Non-goals

Do **not** redesign LIRE based on DiSPQ's internal mechanisms.

DiSPQ is relevant only as the source of the evaluation workload contract and output format.

Do not import DiSPQ-specific implementation ideas such as:

- Active / Sealed / Main route structure
- Balanced Merge publication protocol
- DiSPQ-specific prepare-and-publish design
- DiSPQ version semantics
- DiSPQ merge scheduling
- DiSPQ rebuild mechanisms
- DiSPQ synchronization policy

The internal implementation of concurrent LIRE should be derived from SPFresh's own architecture and semantics.

Also do not implement:

- deletion
- SPDK
- RocksDB
- GPU execution
- replicas beyond the local reproduction's existing semantics
- a complete SPANN serving stack
- optimizations that go beyond what is reasonable for the official SPFresh CPU execution model

The workload evaluated in this project is insert-only after the initial build.

---

## 3. Current Local Implementation

The current local implementation is centered on:

- `src/faiss_ivfflat_lire.h`
- `src/faiss_ivfflat_lire.cpp`
- `src/faiss_baseline_run_eval_common.h`
- `src/faiss_ivfflat_lire_run_eval.cpp`

The LIRE layer currently contains the main algorithmic pieces:

- IVF coarse assignment
- oversized-list detection
- local K=2 split
- balance-aware split clustering
- neighbor-list discovery
- split-child reassignment checks
- neighboring-list reassignment checks
- exact reassignment
- cascading split detection
- membership tracking
- active centroid management

However, its execution model is currently synchronous.

In particular, `IndexIVFFlatLIRE::add_with_ids()` currently acquires a global write lock and performs:

1. insert assignment
2. list append
3. `run_maintenance(...)`
4. split / reassignment
5. quantizer rebuild
6. invariant validation
7. return

while `search()` takes a shared lock on the same global state lock.

As a result:

- Search cannot overlap with insert/maintenance.
- Maintenance is performed synchronously in the insert path.
- A long split/reassignment blocks all concurrent queries.
- The current local implementation does not reflect the official SPFresh background-maintenance execution model.

---

## 4. Official SPFresh Behavior to Follow

The redesign must take the official code as the primary implementation reference.

### 4.1 Foreground insertion is parallel

The official SPFresh update driver uses multiple insert threads.

Relevant official configuration includes:

- `InsertThreadNum`

The update path dispatches insert work across multiple frontend threads.

The local implementation should therefore support multiple insert workers, subject to a controlled CPU-thread budget.

### 4.2 Maintenance is asynchronous

Official `ExtraDynamicSearcher` defines asynchronous maintenance jobs including:

- `SplitAsyncJob`
- `ReassignAsyncJob`
- `MergeAsyncJob`

and maintains background thread pools.

Relevant official configuration includes:

- `AppendThreadNum`
- `ReassignThreadNum`

The local implementation should reproduce the same architectural property:

> Insert should trigger/enqueue LIRE maintenance rather than synchronously completing all maintenance before returning.

It is not necessary to reproduce SPDK- or RocksDB-specific execution details.

### 4.3 Search can run during update

The official configuration includes:

- `SearchDuringUpdate`

and the official update driver executes searches while the asynchronous insertion/update task is still running.

Therefore the local concurrent evaluator must support:

> Search overlapping with foreground insert and background LIRE maintenance.

### 4.4 Fine-grained synchronization is expected

The official implementation uses fine-grained synchronization around posting operations and maintenance coordination, including:

- fine-grained RW locking
- posting state tracking
- active maintenance job tracking
- concurrent maintenance queues/thread pools

The local implementation should move away from a single long-lived global write lock covering the full maintenance path.

However, do not blindly copy storage-specific locking structures from SPFresh. The local implementation should adapt these semantics appropriately to Faiss inverted lists.

---

## 5. Target Execution Model

The target architecture should be:

```text
                         +------------------+
                         | Query Worker 0   |
                         +------------------+
                         | Query Worker 1   |
                         +------------------+
                         | ...              |
                         +------------------+

Insert Dispatcher / Insert Workers
        |
        +--> coarse assignment
        +--> posting append
        +--> detect oversized postings
        +--> enqueue LIRE maintenance jobs
        +--> return

                         +----------------------+
                         | Maintenance Queue    |
                         +----------------------+
                            |       |       |
                            v       v       v
                         Worker0 Worker1 Worker2
                            |       |       |
                         Split / Reassign / cascading maintenance
```

The essential concurrency relationship is:

```text
Query || Insert || LIRE Maintenance
```

This should be a real overlap, not just alternating phases.

---

## 6. Synchronization Design Requirements

The synchronization design should follow the official SPFresh principle of local posting synchronization.

### 6.1 Per-list synchronization

Introduce per-physical-list synchronization.

A reasonable Faiss-side representation is one lock per physical IVF list.

Expected semantics:

- Query scanning a list: shared/read access
- Insert appending to a list: exclusive/write access
- Maintenance reading a list snapshot: shared/read access
- Maintenance replacing/mutating a list: exclusive/write access

This allows unrelated lists to proceed concurrently.

Example:

```text
Query probes C1, C2, C3
Insert appends to C50
```

These operations should not serialize on one global index lock.

### 6.2 Lock ordering

Any maintenance operation that needs to modify multiple lists must acquire their locks in a deterministic order, e.g. ascending list ID.

This is required to prevent deadlock among multiple maintenance workers.

### 6.3 Shared global metadata

Some state remains global, including:

- active-list bitmap/state
- free list slots
- active centroid mapping
- global membership metadata
- aggregate counters

Use a short-lived global coordination lock only where necessary.

Do not hold this global lock during expensive split/reassignment computation.

---

## 7. Insert Path Redesign

The current insert path performs maintenance synchronously.

Change it so that the foreground insert path performs only:

1. assign incoming vectors to current active centroids
2. group vectors by target posting list
3. append vectors to target lists
4. update membership metadata
5. update committed/visible insert frontier
6. detect oversized touched lists
7. enqueue maintenance jobs for those lists
8. return

Do not execute the complete split/reassign cascade synchronously inside `add_with_ids()`.

Foreground insert latency should represent insert work, not total maintenance completion time.

---

## 8. Maintenance Worker Model

Introduce a maintenance queue and a fixed-size maintenance worker pool.

The local implementation does not need to reproduce every official SPFresh queue separately.

A single shared maintenance-job abstraction is acceptable initially.

A maintenance job should identify at least:

- source list / posting ID
- triggering state/generation
- optional diagnostic metadata

A worker should:

1. re-check that the list still needs maintenance
2. snapshot the required posting and centroid state
3. perform split clustering
4. identify neighboring lists
5. evaluate reassignment candidates
6. prepare replacement list contents
7. validate that relevant state has not changed incompatibly
8. commit the maintenance result
9. detect any newly oversized affected lists
10. enqueue cascading maintenance jobs if needed

The worker must not hold broad/global locks during expensive CPU computation.

---

## 9. Generation / Staleness Protection

Because maintenance computation will occur outside exclusive locks, the implementation must detect stale snapshots.

Add generation/version counters for mutable list state.

For example:

```text
snapshot:
    list C5 generation = 14

later commit:
    if generation(C5) != 14:
        discard/retry/requeue
```

The same applies to affected neighboring lists whose contents are part of the prepared maintenance result.

A stale maintenance result must never overwrite a newer foreground insert.

---

## 10. Split / Reassign Semantics Must Stay LIRE-Compatible

Do not change the algorithmic conditions already reproduced from SPFresh.

Preserve:

- oversized-list trigger semantics
- K=2 split
- balance-aware clustering behavior
- local split-child reassignment condition
- neighboring-list reassignment condition
- exact target reassignment
- cascading split behavior
- `lire_reassign_k`
- `lire_split_threshold`
- `lire_enable_reassign`

The redesign is primarily about execution architecture and synchronization.

Do not replace the existing LIRE algorithm with a new partition-balancing method.

---

## 11. Centroid / Quantizer State

The current code rebuilds the active centroid quantizer after maintenance.

This must become safe under concurrent query execution.

The key requirement is:

> A query must never observe a partially rebuilt centroid-to-list mapping.

Use a concurrency-safe centroid state representation.

An immutable active-centroid snapshot is acceptable if implemented purely as a safe way to expose Faiss centroid search state.

The design should remain a Faiss-based CPU implementation.

Do not introduce a new ANN structure for centroid search.

Preferred backend remains:

- Faiss `IndexFlatL2`
- or equivalent Faiss CPU primitive

---

## 12. Search Path

Concurrent search should:

1. obtain a stable view of current visible/committed insert frontier
2. obtain a stable active-centroid routing state
3. route to `nprobe` active lists
4. scan the corresponding lists under read synchronization
5. return top-k IDs
6. record latency, completion time, and the logical insert frontier associated with the query

The search implementation should avoid a whole-index exclusive/shared lock that serializes all query/update activity.

---

## 13. Visibility and GT Compatibility

The evaluator must be compatible with DiSPQ-generated GT files.

This is an **evaluation compatibility requirement**, not an instruction to copy DiSPQ's internal concurrency implementation.

The local concurrent evaluator should define a clear committed insertion frontier aligned with dynamic insertion batches.

For a batch:

```text
[begin_row, end_row)
```

the evaluator should publish the batch only when its foreground insert operation has successfully completed.

The query record must be associated with the corresponding committed frontier.

Maintenance does not change logical dataset membership and therefore must not advance the GT frontier.

---

## 14. DiSPQ GT Reader

Add support for directly loading the GT artifact generated by the DiSPQ concurrent evaluator.

Do not recompute exact GT inside the LIRE baseline for formal concurrent experiments.

The reader should validate all available compatibility metadata, including at least:

- file magic/version
- query count
- top-k
- initial rows
- batch size
- number of frontiers
- each stored frontier

Then use:

```text
query committed frontier
        ->
matching GT frontier
        ->
expected top-k
```

to compute Recall@k.

Do not silently map incompatible GT files.

---

## 15. Concurrent Evaluation Output

The concurrent LIRE evaluator should produce outputs aligned with the DiSPQ concurrent evaluator where the metrics have equivalent meaning.

Target outputs:

```text
summary.json
*_timeseries.csv
*_insert.csv
*_events.csv
*_progress.csv
```

The output format does not need to include DiSPQ-only fields whose semantics do not exist in LIRE.

### Required query metrics

At minimum:

- query count
- QPS
- mean latency
- p50 latency
- p95 latency
- p99 latency
- max latency
- Recall@k

### Required insert metrics

At minimum:

- batch ID
- begin/end row
- start/end time
- insert latency
- effective insert throughput

### Required LIRE maintenance metrics

At minimum, enough information to understand maintenance overlap:

- maintenance job ID
- triggering list
- start time
- compute completion time
- commit start / commit-wait timing if applicable
- commit completion time
- end time
- number of affected lists
- split count
- reassigned vector count

Exact schema may be adapted to the implementation.

---

## 16. Dynamic Insert Progress

The main paper-facing x-axis is:

```text
Dynamic Insert Progress (%)
```

The first completed dynamic insertion batch is 0%.

The final completed dynamic insertion batch is 100%.

Use actual inserted-row frontiers rather than wall-clock time for the x-coordinate.

Wall-clock timestamps should still be retained for:

- QPS calculation
- latency measurement
- event duration
- debugging

---

## 17. Runtime Truncation

The concurrent LIRE evaluator should support the same runtime tuning concept:

```text
--max-insert-vectors N
```

This is not a config field.

If omitted:

- run the complete dynamic insertion stream

If provided:

- stop the dynamic stream after at most N inserted vectors after initial build
- drain maintenance normally
- stop query workers normally
- save all metrics normally
- use the corresponding truncated GT frontier
- still map the truncated run's progress to 0%-100%

This must be implemented as a shortened workload, not an abnormal abort.

---

## 18. CPU Optimization Policy

The implementation should make reasonable use of CPU optimizations present in official SPFresh or standard Faiss CPU execution.

The project explicitly describes this baseline as:

> Faiss-based reproduction of LIRE

Therefore it is desirable to use Faiss's optimized CPU kernels.

### Allowed

- Faiss CPU SIMD kernels
- AVX2 / AVX512 Faiss builds where supported by the experiment machine
- OpenMP
- request-level worker parallelism
- background maintenance worker parallelism
- batching distance calculations
- replacing scalar distance loops with Faiss distance primitives
- reducing unnecessary repeated data copies
- reusing Faiss `IndexFlatL2` or Faiss distance functions for centroid distance computation

### Not allowed

Do not introduce algorithmic improvements beyond the official baseline, such as:

- new approximate reassign algorithms
- new pruning schemes
- different split objectives
- new centroid indexes not used to simply reproduce existing exact routing behavior
- GPU acceleration
- DiSPQ-specific balancing ideas
- DiSPQ-specific merge/publication algorithms

---

## 19. AVX / Faiss Requirements

The current project uses:

```cmake
FAISS_OPT_LEVEL generic
```

Review this and make the build capable of using the appropriate optimized Faiss CPU target supported by the experiment machine.

Prefer a configurable build-time setting such as AVX2 / AVX512 rather than hard-coding assumptions that may fail on another machine.

Where local code currently uses a scalar L2 loop, prefer a Faiss-provided CPU distance primitive such as the appropriate Faiss L2 routine rather than handwritten AVX intrinsics.

Priority targets include:

- centroid distance
- nearest-active-centroid computation
- nearest-neighbor centroid selection
- split assignment
- reassignment filtering
- candidate-to-centroid distance computation

The goal is to benefit from Faiss's SIMD implementation while keeping LIRE's algorithm unchanged.

---

## 20. OpenMP / Thread-Budget Rules

Avoid nested oversubscription.

For the concurrent workload, prefer:

```text
request-level query workers
+
foreground insert workers
+
maintenance workers
```

over spawning a full OpenMP team inside every individual query.

A query worker should normally use a single request thread unless there is a specific existing Faiss mechanism whose thread use is explicitly budgeted.

The total concurrent CPU demand should be configurable and understandable.

Do not accidentally create:

```text
query_workers * omp_threads_per_query
```

large nested thread counts.

---

## 21. No Deletion Work

Deletion is outside the scope of this redesign.

Do not implement:

- delete API behavior
- tombstone/version deletion logic
- delete-aware GT
- delete-aware membership cleanup
- MSTuring insert-delete execution

The evaluated workloads are insert-only.

---

## 22. Suggested Implementation Phases

Do not implement everything in one uncontrolled patch.

### Phase 1: Concurrent execution structure

Implement:

- query worker pool
- insert worker/dispatcher
- maintenance queue
- maintenance worker pool
- clean startup/shutdown
- drain maintenance on workload completion

Do not change the LIRE algorithm yet.

### Phase 2: Correct fine-grained synchronization

Implement:

- per-list synchronization
- shared global metadata coordination
- generation counters
- safe centroid state
- stale maintenance detection
- deadlock-safe multi-list lock ordering

Verify that concurrent search never observes torn state.

### Phase 3: Evaluation compatibility

Implement:

- DiSPQ GT reader
- query records
- insert records
- maintenance events
- progress CSV
- Recall curve
- `--max-insert-vectors`
- compatible summary and plotting inputs

### Phase 4: CPU optimization

After correctness is established, optimize:

- scalar L2 -> Faiss SIMD distance routine
- centroid distance batching
- split computation parallelism
- neighbor scan parallelism
- reassignment batch distance
- redundant list copies
- Faiss AVX build configuration

All Phase 4 changes must preserve the result semantics of Phase 3.

---

## 23. Validation Expectations

Do not add standalone test files unless explicitly requested.

Instead, add lightweight runtime validation or debug-only checks where useful.

At minimum verify:

- every committed visible ID appears exactly once
- membership metadata matches list placement
- inactive lists are not scanned as active
- no query returns IDs beyond its visible insertion frontier
- split/reassign publication never creates torn list/centroid state
- stale maintenance jobs do not overwrite newer inserts
- maintenance queue drains successfully at shutdown
- no deadlock under concurrent Query/Insert/Maintenance
- sequential LIRE and concurrent LIRE reach equivalent logical contents after maintenance drains

---

## 24. Files and Refactoring Scope

Prefer a clean separation between:

1. LIRE index logic
2. maintenance scheduling
3. concurrent evaluator
4. GT reader
5. metrics/output

The current `faiss_baseline_run_eval_common.h` is already very large.

Avoid adding all concurrency logic into that file.

Reasonable new files may include:

```text
src/lire_maintenance.h
src/lire_maintenance.cpp
src/dispq_gt_reader.h
src/dispq_gt_reader.cpp
src/faiss_lire_concurrent_eval.cpp
src/concurrent_eval_metrics.h
src/concurrent_eval_metrics.cpp
```

Exact names may differ.

Keep the existing sequential evaluator functional where practical.

---

## 25. Final Design Constraint

When there is ambiguity, use this priority order:

1. official SPFresh code behavior
2. SPFresh/LIRE paper semantics
3. existing local LIRE algorithm
4. Faiss-supported CPU implementation
5. DiSPQ evaluation I/O compatibility

Do **not** reverse this order.

In particular:

> DiSPQ should define only the experimental interface and GT/output compatibility, not how LIRE internally implements concurrency.

---

## 26. Deliverables for the First Coding Pass

Before making large changes, inspect the official SPFresh source and the local repository and report:

1. exact official code paths used as references
2. current local code paths that need modification
3. proposed classes/data structures for:
   - maintenance queue
   - maintenance worker pool
   - per-list synchronization
   - centroid state
   - query/insert records
4. lock hierarchy and lock ordering
5. maintenance-job lifecycle
6. GT mapping strategy
7. thread-budget strategy
8. exact files expected to be added or modified

Then implement Phase 1 and Phase 2 first.

Do not begin aggressive CPU optimization until the concurrent semantics are correct and measurable.
