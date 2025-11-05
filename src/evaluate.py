#!/usr/bin/env python3
"""
Evaluate simulation outputs and compute tables for the paper.

Outputs:
  results/summary_by_strategy_workload.csv
  results/best_by_workload.csv
  results/load_balance.csv
  results/summary_report.txt
"""

import os
import glob
import pandas as pd
import numpy as np

RESULTS_DIR = "results"
PARTITIONS_DIR = "partitions"
OUT_SUMMARY = os.path.join(RESULTS_DIR, "summary_by_strategy_workload.csv")
OUT_BEST = os.path.join(RESULTS_DIR, "best_by_workload.csv")
OUT_LOAD = os.path.join(RESULTS_DIR, "load_balance.csv")
OUT_REPORT = os.path.join(RESULTS_DIR, "summary_report.txt")

# Strategy -> partitions parquet filename
PARTITION_FILES = {
    "round_robin": os.path.join(PARTITIONS_DIR, "round_robin_P4.parquet"),
    "hash_user":   os.path.join(PARTITIONS_DIR, "hash_user_P4.parquet"),
    "range_pop":   os.path.join(PARTITIONS_DIR, "range_pop_P4.parquet"),
    "group_book":  os.path.join(PARTITIONS_DIR, "group_book_P4.parquet"),
}

def gini_coefficient(values: np.ndarray) -> float:
    """Gini of non-negative values."""
    x = np.array(values, dtype=float)
    if np.any(x < 0):
        raise ValueError("Gini requires non-negative values")
    if np.all(x == 0):
        return 0.0
    x.sort()
    n = len(x)
    cumx = np.cumsum(x)
    # Gini = 1 + (1/n) - 2 * sum((n+1-i) * x_i) / (n * sum(x))
    g = 1 + 1/n - 2 * np.sum(cumx) / (n * cumx[-1])
    return float(g)

def aggregate_logs() -> pd.DataFrame:
    files = glob.glob(os.path.join(RESULTS_DIR, "log_*_P4_*.csv"))
    if not files:
        raise SystemExit("No log_*.csv files found in results/. Run simulate.py first.")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Basic sanity
    needed = {"strategy", "workload", "P", "partitions_touched", "rows_scanned", "rows_matched", "latency_ms"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in logs: {missing}")

    grouped = df.groupby(["strategy", "workload"], as_index=False).agg(
        queries=("latency_ms", "size"),
        mean_latency_ms=("latency_ms", "mean"),
        median_latency_ms=("latency_ms", "median"),
        mean_partitions_touched=("partitions_touched", "mean"),
        mean_rows_scanned=("rows_scanned", "mean"),
        mean_rows_matched=("rows_matched", "mean"),
    )
    # Sort for readability
    grouped = grouped.sort_values(["workload", "mean_latency_ms"]).reset_index(drop=True)
    grouped.to_csv(OUT_SUMMARY, index=False)
    return grouped

def rank_best_by_workload(summary_df: pd.DataFrame) -> pd.DataFrame:
    best = (
        summary_df.sort_values(["workload", "mean_latency_ms"])
        .groupby("workload", as_index=False)
        .first()[["workload", "strategy", "mean_latency_ms", "mean_partitions_touched"]]
    )
    best.rename(columns={
        "strategy": "best_strategy",
        "mean_latency_ms": "best_mean_latency_ms",
        "mean_partitions_touched": "best_mean_partitions_touched"
    }, inplace=True)
    best.to_csv(OUT_BEST, index=False)
    return best

def compute_load_balance() -> pd.DataFrame:
    rows = []
    for strat, path in PARTITION_FILES.items():
        if not os.path.exists(path):
            # Skip if not produced
            continue
        df = pd.read_parquet(path, columns=["part"])
        counts = df["part"].value_counts().sort_index()
        parts = counts.index.tolist()
        sizes = counts.values.astype(float)
        total = sizes.sum()
        share = (sizes / total).round(6)
        gini = gini_coefficient(sizes)
        for p, c, s in zip(parts, sizes, share):
            rows.append({"strategy": strat, "part": int(p), "rows": int(c), "share": float(s), "gini": gini})
    lb = pd.DataFrame(rows)
    lb.to_csv(OUT_LOAD, index=False)
    return lb

def write_report(summary_df: pd.DataFrame, best_df: pd.DataFrame, lb_df: pd.DataFrame):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_REPORT, "w") as f:
        f.write("=== Evaluation Summary (P=4) ===\n\n")

        # Per-workload top choice
        f.write("Best strategy per workload (by mean latency):\n")
        for _, r in best_df.iterrows():
            f.write(
                f"  - {r['workload']}: {r['best_strategy']}  "
                f"(mean latency = {r['best_mean_latency_ms']:.3f} ms; "
                f"avg partitions touched = {r['best_mean_partitions_touched']:.2f})\n"
            )
        f.write("\n")

        # Brief note on load balance
        f.write("Load balance (Gini; lower is more balanced):\n")
        for strat in lb_df["strategy"].unique():
            g = lb_df[lb_df["strategy"] == strat]["gini"].iloc[0]
            f.write(f"  - {strat}: Gini = {g:.4f}\n")

        f.write("\nTables written:\n")
        f.write(f"  - {OUT_SUMMARY}\n")
        f.write(f"  - {OUT_BEST}\n")
        f.write(f"  - {OUT_LOAD}\n")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = aggregate_logs()
    best = rank_best_by_workload(summary)
    lb = compute_load_balance()
    write_report(summary, best, lb)
    print(f"[OK] Wrote:\n  {OUT_SUMMARY}\n  {OUT_BEST}\n  {OUT_LOAD}\n  {OUT_REPORT}")

if __name__ == "__main__":
    main()
