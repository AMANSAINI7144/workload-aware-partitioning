#!/usr/bin/env python3
"""
Simulate running a workload on a given partitioned layout.
Inputs:
  --strategy: round_robin | hash_user | range_pop | group_book
  --workload: W1 | W2 | W3
  --P: partitions (default 4)
Reads the partitioned parquet from ./partitions/*_P{P}.parquet
Outputs per-query CSV logs in ./results/
"""

import os, json, argparse, numpy as np, pandas as pd
from time import perf_counter

# Cost model constants (tune as needed)
SCAN_MS_PER_1K = 0.05
NET_MS_PER_EXTRA_PART = 2.0

def load_workload(path):
    with open(path, "r") as f:
        return json.load(f)

def calc_latency(rows_scanned, parts_touched):
    scan_ms = SCAN_MS_PER_1K * (rows_scanned / 1000.0)
    net_ms  = NET_MS_PER_EXTRA_PART * max(0, parts_touched - 1)
    return scan_ms + net_ms

def prepare_partition_frames(df):
    """Return list of per-partition dataframes and metadata per partition."""
    parts = sorted(df["part"].unique().tolist())
    frames = [df[df["part"] == p].copy() for p in parts]
    # per-partition min/max popularity to prune for W2
    meta = []
    for f in frames:
        if "popularity" in f.columns and len(f) > 0:
            meta.append({"min_pop": float(f["popularity"].min()),
                         "max_pop": float(f["popularity"].max())})
        else:
            meta.append({"min_pop": None, "max_pop": None})
    return frames, meta

def run_W1(frames, workload_spec):
    """User lookups: count partitions touched and rows."""
    # Build index: user -> partitions that contain this user
    user_to_parts = {}
    for p, f in enumerate(frames):
        u = f["user_idx"].unique()
        for uid in u:
            user_to_parts.setdefault(int(uid), set()).add(p)

    logs = []
    for uid in workload_spec["users"]:
        parts = user_to_parts.get(int(uid), set())
        parts_touched = len(parts)
        rows_scanned = 0
        rows_matched = 0
        # scan only those partitions
        for p in parts:
            ff = frames[p]
            sel = ff[ff["user_idx"] == uid]
            rows_matched += len(sel)
            rows_scanned += len(sel)  # assume indexed lookup → scan≈match
        latency = calc_latency(rows_scanned, parts_touched)
        logs.append((parts_touched, rows_scanned, rows_matched, latency))
    return logs

def run_W2(frames, meta, workload_spec):
    """Range-scan over popularity."""
    logs = []
    for rg in workload_spec["ranges"]:
        lo, hi = rg["lo"], rg["hi"]
        parts_touch = []
        rows_scanned = 0
        rows_matched = 0
        # prune partitions by min/max popularity
        for p, (f, m) in enumerate(zip(frames, meta)):
            if m["min_pop"] is None:
                continue
            if m["max_pop"] < lo or m["min_pop"] > hi:
                continue  # partition fully outside
            parts_touch.append(p)
            sel = f[(f["popularity"] >= lo) & (f["popularity"] <= hi)]
            rows_scanned += len(sel)   # scan only qualifying rows (optimistic)
            rows_matched += len(sel)
        latency = calc_latency(rows_scanned, len(parts_touch))
        logs.append((len(parts_touch), rows_scanned, rows_matched, latency))
    return logs

def run_W3(frames, workload_spec):
    """Bundle queries over book_idx sets."""
    # Build book -> part map (assume each book resides in exactly one part per layout)
    book_to_part = {}
    for p, f in enumerate(frames):
        for b in f["book_idx"].unique():
            book_to_part[int(b)] = p

    logs = []
    for bset in workload_spec["bundles"]:
        books = [int(b) for b in bset["book_idxs"]]
        parts = set()
        for b in books:
            parts.add(book_to_part.get(b, -1))
        parts.discard(-1)
        rows_scanned = 0
        rows_matched = 0
        for p in parts:
            ff = frames[p]
            sel = ff[ff["book_idx"].isin(books)]
            rows_scanned += len(sel)
            rows_matched += len(sel)
        latency = calc_latency(rows_scanned, len(parts))
        logs.append((len(parts), rows_scanned, rows_matched, latency))
    return logs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True,
                    choices=["round_robin","hash_user","range_pop","group_book"])
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--workload", required=True, choices=["W1","W2","W3"])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    # Map strategy -> partition file
    part_path = {
        "round_robin": f"partitions/round_robin_P{args.P}.parquet",
        "hash_user":   f"partitions/hash_user_P{args.P}.parquet",
        "range_pop":   f"partitions/range_pop_P{args.P}.parquet",
        "group_book":  f"partitions/group_book_P{args.P}.parquet",
    }[args.strategy]

    # Map workload -> json file
    wl_path = {
        "W1": "workloads/W1_user_lookup.json",
        "W2": "workloads/W2_range_popularity.json",
        "W3": "workloads/W3_book_bundles.json",
    }[args.workload]

    if not os.path.exists(part_path):
        raise SystemExit(f"Partition file not found: {part_path}")
    if not os.path.exists(wl_path):
        raise SystemExit(f"Workload file not found: {wl_path} (run workloads.py first)")

    df = pd.read_parquet(part_path)
    frames, meta = prepare_partition_frames(df)
    wl = load_workload(wl_path)

    t0 = perf_counter()
    if args.workload == "W1":
        logs = run_W1(frames, wl)
    elif args.workload == "W2":
        logs = run_W2(frames, meta, wl)
    else:
        logs = run_W3(frames, wl)
    t1 = perf_counter()

    # write logs
    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, f"log_{args.strategy}_P{args.P}_{args.workload}.csv")
    out_df = pd.DataFrame(logs, columns=["partitions_touched","rows_scanned","rows_matched","latency_ms"])
    out_df.insert(0, "workload", args.workload)
    out_df.insert(0, "P", args.P)
    out_df.insert(0, "strategy", args.strategy)
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} ({len(out_df)} queries) in {t1 - t0:.2f}s")

if __name__ == "__main__":
    main()
