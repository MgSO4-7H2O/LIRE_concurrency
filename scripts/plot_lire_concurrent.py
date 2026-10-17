#!/usr/bin/env python3
"""Plot a concurrent LIRE run as a four-panel progress timeline."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key, default=float("nan")):
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_output(summary_path, summary, key, suffix):
    configured = summary.get("outputs", {}).get(key)
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path
    return summary_path.with_name(summary_path.stem + suffix)


def progress_at_time(time_ms, progress_rows, timeseries_rows):
    """Map a maintenance timestamp onto the dynamic-insert progress axis."""
    anchors_t = []
    anchors_p = []
    previous_progress = 0.0
    for row in progress_rows:
        start = number(row, "batch_start_ms")
        end = number(row, "batch_end_ms")
        progress = number(row, "dynamic_insert_progress_pct")
        if np.isfinite(start) and np.isfinite(progress):
            anchors_t.append(start)
            anchors_p.append(previous_progress)
        if np.isfinite(end) and np.isfinite(progress):
            anchors_t.append(end)
            anchors_p.append(progress)
            previous_progress = progress
    if len(anchors_t) >= 2:
        order = np.argsort(anchors_t)
        times = np.asarray(anchors_t)[order]
        values = np.asarray(anchors_p)[order]
        return float(np.interp(time_ms, times, values))

    times = [number(row, "time_ms") for row in timeseries_rows]
    progress = [number(row, "dynamic_insert_progress_pct") for row in timeseries_rows]
    valid = [(t, p) for t, p in zip(times, progress) if np.isfinite(t) and np.isfinite(p)]
    if not valid:
        return 0.0
    return float(np.interp(time_ms, [v[0] for v in valid], [v[1] for v in valid]))


def make_title(summary, summary_path):
    workers = summary.get("query_workers")
    per_request = summary.get("query_threads_per_request")
    insert_threads = summary.get("insert_threads")
    maintenance_threads = summary.get("maintenance_threads")
    target = summary.get("insert_rate_limit_vecps")
    if target is None:
        target = summary.get("concurrent_target_insert_vecps")
    if workers is not None:
        query = f"Q{workers}"
        if per_request is not None:
            query += f"x{per_request}"
        pieces = [query]
        if insert_threads is not None:
            pieces.append(f"I{insert_threads}")
        if maintenance_threads is not None:
            pieces.append(f"M{maintenance_threads}")
        if target is not None:
            pieces.append(f"target={target:g}")
        return " / ".join(pieces)
    return summary_path.stem


def aggregate_timeseries(rows):
    """Collapse repeated 200 ms buckets at the same published frontier."""
    groups = {}
    for row in rows:
        progress = number(row, "dynamic_insert_progress_pct")
        if not np.isfinite(progress):
            continue
        group = groups.setdefault(progress, [])
        group.append(row)

    out = []
    for progress, group in sorted(groups.items()):
        counts = [max(0.0, number(row, "query_count", 0.0)) for row in group]
        total_count = sum(counts)

        def weighted_mean(key):
            values = [number(row, key) for row in group]
            valid = [(v, w) for v, w in zip(values, counts) if np.isfinite(v) and w > 0]
            denom = sum(w for _, w in valid)
            return sum(v * w for v, w in valid) / denom if denom else float("nan")

        def bucket_mean(key):
            values = [number(row, key) for row in group]
            values = [value for value in values if np.isfinite(value)]
            return float(np.mean(values)) if values else float("nan")

        out.append({
            "progress": progress,
            "recall": weighted_mean("recall_at_k"),
            # QPS and latency percentiles are bucket metrics; average the buckets
            # so a busy bucket does not dominate the throughput curve.
            "qps": bucket_mean("qps"),
            "p50": bucket_mean("p50_latency_ms"),
            "p95": bucket_mean("p95_latency_ms"),
            "p99": bucket_mean("p99_latency_ms"),
            "query_count": total_count,
        })
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot recall, QPS, query latency, and insertion latency for a LIRE run."
    )
    parser.add_argument("summary_json", type=Path, help="LIRE concurrent summary JSON")
    parser.add_argument("-o", "--output", type=Path,
                        help="output image path (default: <summary>_query_summary_timeline.png)")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    summary_path = args.summary_json
    summary = json.loads(summary_path.read_text())
    timeseries_path = resolve_output(
        summary_path, summary, "timeseries", "_timeseries.csv")
    progress_path = resolve_output(
        summary_path, summary, "progress", "_progress.csv")
    events_path = resolve_output(
        summary_path, summary, "events", "_events.csv")

    timeseries = read_csv(timeseries_path)
    progress = read_csv(progress_path) if progress_path.exists() else []
    events = read_csv(events_path) if events_path.exists() else []
    if not timeseries:
        raise SystemExit(f"No time-series rows found in {timeseries_path}")

    timeline = aggregate_timeseries(timeseries)
    x = np.asarray([row["progress"] for row in timeline])
    recall = np.asarray([row["recall"] for row in timeline])
    qps = np.asarray([row["qps"] for row in timeline])
    p50 = np.asarray([row["p50"] for row in timeline])
    p95 = np.asarray([row["p95"] for row in timeline])
    p99 = np.asarray([row["p99"] for row in timeline])

    batch_x = np.asarray([number(row, "dynamic_insert_progress_pct") for row in progress])
    batch_ms = np.asarray([
        number(row, "batch_end_ms") - number(row, "batch_start_ms")
        for row in progress
    ])
    insert_ms = np.asarray([number(row, "insert_latency_ms") for row in progress])

    output_path = args.output or summary_path.with_name(
        summary_path.stem + "_query_summary_timeline.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    fig.subplots_adjust(hspace=0.08, top=0.91, bottom=0.09, left=0.09, right=0.98)
    fig.suptitle(make_title(summary, summary_path), fontsize=15, y=0.97)

    purple = "#9467bd"
    blue = "#2585bd"
    orange = "#ff7f0e"
    green = "#2ca02c"
    pale_green = "#77c679"

    for ax in axes:
        ax.grid(True, color="#b0b0b0", alpha=0.28, linewidth=0.8)
        ax.set_xlim(0, 100)

    axes[0].plot(x, recall, color=purple, marker="o", markersize=3.8,
                 linewidth=1.4, label="Recall@k")
    axes[0].set_ylabel("Recall@k")
    axes[0].legend(loc="upper left", framealpha=0.9)

    axes[1].plot(x, qps, color=blue, marker="o", markersize=3.2,
                 linewidth=1.25, label="Query QPS")
    axes[1].set_ylabel("Query QPS")
    axes[1].legend(loc="upper left", framealpha=0.9)

    axes[2].plot(x, p50, color=blue, linewidth=1.7, label="p50")
    axes[2].plot(x, p95, color=orange, linewidth=1.7, label="p95")
    axes[2].plot(x, p99, color=green, linewidth=1.7, label="p99")
    axes[2].set_ylabel("Query latency (ms)")
    axes[2].legend(loc="upper left", framealpha=0.9)

    if len(batch_x):
        axes[3].plot(batch_x, batch_ms, color=blue, marker="o", markersize=3.6,
                     linewidth=1.3, label="Batch duration")
        axes[3].plot(batch_x, insert_ms, color=orange, marker="o", markersize=3.2,
                     linewidth=1.2, linestyle="--", label="Insert latency")
    axes[3].set_ylabel("Insert latency (ms)")
    axes[3].set_xlabel("Dynamic Insert Progress (%)")
    axes[3].legend(loc="upper left", framealpha=0.9)

    # Shade maintenance jobs and mark their completion on the progress axis.
    event_windows = 0
    completion_marks = 0
    for event in events:
        start = number(event, "request_ms", number(event, "start_ms"))
        end = number(event, "end_ms", start)
        if not np.isfinite(start):
            continue
        start_x = progress_at_time(start, progress, timeseries)
        end_x = progress_at_time(end, progress, timeseries) if np.isfinite(end) else start_x
        if end_x > start_x:
            for ax in axes:
                ax.axvspan(start_x, end_x, color=pale_green, alpha=0.16, linewidth=0)
            event_windows += 1
        commit = number(event, "commit_done_ms")
        if np.isfinite(commit):
            commit_x = progress_at_time(commit, progress, timeseries)
            for ax in axes:
                ax.axvline(commit_x, color="#16802b", linestyle="--",
                           linewidth=0.85, alpha=0.72)
            completion_marks += 1

    # Re-add plotted data above the maintenance shading.
    for ax in axes:
        ax.set_axisbelow(True)

    if event_windows or completion_marks:
        recall_legend = axes[0].legend(loc="upper left", framealpha=0.9)
        axes[0].add_artist(recall_legend)
        handles = [
            Patch(facecolor=pale_green, alpha=0.22, label="LIRE maintenance"),
            Line2D([0], [0], color="#16802b", linestyle="--",
                   linewidth=1.1, label="Maintenance completion"),
        ]
        axes[0].legend(handles=handles, loc="upper right", framealpha=0.9)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
