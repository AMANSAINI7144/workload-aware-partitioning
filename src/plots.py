#!/usr/bin/env python3
"""
Make plots for the paper & slides from evaluation CSVs.

Inputs:
  results/summary_by_strategy_workload.csv
  results/load_balance.csv

Outputs (in figs/):
  latency_by_strategy_W1.png
  latency_by_strategy_W2.png
  latency_by_strategy_W3.png
  parts_touched_by_strategy_W1.png
  parts_touched_by_strategy_W2.png
  parts_touched_by_strategy_W3.png
  load_balance_round_robin.png
  load_balance_hash_user.png
  load_balance_range_pop.png
  load_balance_group_book.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIGS_DIR = "figs"

def bar_plot(ax, xlabels, heights, title, ylabel):
    ax.bar(range(len(xlabels)), heights)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

def plot_latency_per_workload(df):
    os.makedirs(FIGS_DIR, exist_ok=True)
    for wl in sorted(df["workload"].unique()):
        sub = df[df["workload"] == wl].sort_values("mean_latency_ms")
        x = sub["strategy"].tolist()
        y = sub["mean_latency_ms"].tolist()
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_plot(ax, x, y, title=f"Mean Latency by Strategy — {wl}", ylabel="Latency (ms)")
        out = os.path.join(FIGS_DIR, f"latency_by_strategy_{wl}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)

def plot_parts_touched_per_workload(df):
    for wl in sorted(df["workload"].unique()):
        sub = df[df["workload"] == wl].sort_values("mean_partitions_touched")
        x = sub["strategy"].tolist()
        y = sub["mean_partitions_touched"].tolist()
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_plot(ax, x, y, title=f"Avg Partitions Touched — {wl}", ylabel="Partitions (avg)")
        out = os.path.join(FIGS_DIR, f"parts_touched_by_strategy_{wl}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)

def plot_load_balance(lb_df):
    # One chart per strategy: bars for partition shares
    for strat in ["round_robin", "hash_user", "range_pop", "group_book"]:
        sub = lb_df[lb_df["strategy"] == strat].sort_values("part")
        if sub.empty:
            continue
        parts = sub["part"].tolist()
        shares = sub["share"].tolist()
        gini = sub["gini"].iloc[0]
        fig, ax = plt.subplots(figsize=(6, 4))
        bar_plot(ax, [f"p{p}" for p in parts], shares,
                 title=f"Load Balance — {strat} (Gini={gini:.3f})",
                 ylabel="Share of rows")
        out = os.path.join(FIGS_DIR, f"load_balance_{strat}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)

def main():
    summary_path = os.path.join(RESULTS_DIR, "summary_by_strategy_workload.csv")
    load_path = os.path.join(RESULTS_DIR, "load_balance.csv")
    if not os.path.exists(summary_path):
        raise SystemExit("Missing summary_by_strategy_workload.csv. Run evaluate.py first.")
    if not os.path.exists(load_path):
        raise SystemExit("Missing load_balance.csv. Run evaluate.py first.")

    df = pd.read_csv(summary_path)
    lb = pd.read_csv(load_path)

    # Ensure ordering of workloads
    wl_map = {"W1": 1, "W2": 2, "W3": 3}
    df["wl_order"] = df["workload"].map(wl_map).fillna(99)
    df = df.sort_values(["wl_order", "strategy"])

    plot_latency_per_workload(df)
    plot_parts_touched_per_workload(df)
    plot_load_balance(lb)

    print(f"[OK] Plots saved in {FIGS_DIR}/")

if __name__ == "__main__":
    main()
