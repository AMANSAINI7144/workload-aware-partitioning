#!/usr/bin/env python3
import os, argparse
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/interactions.parquet")
    ap.add_argument("--output", default="partitions/range_pop_P4.parquet")
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--col", default="popularity", help="numeric column to range-partition")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.read_parquet(args.input)
    if args.col not in df.columns:
        raise SystemExit(f"Column '{args.col}' not in dataframe.")

    # Derive a per-book popularity table (robust even if duplicates exist)
    pop = df.groupby("book_idx").size().rename("popularity").reset_index()
    # Quantile labels 0..P-1
    labels = list(range(args.P))
    try:
        pop["part"] = pd.qcut(pop["popularity"], q=args.P, labels=labels, duplicates="drop")
        # If duplicates caused fewer bins, remap to 0..k-1 and warn
        if pop["part"].isna().any() or pop["part"].nunique() < args.P:
            # fallback to rank-based equal-count bins
            ranks = pop["popularity"].rank(method="first")
            pop["part"] = pd.qcut(ranks, q=args.P, labels=labels, duplicates="drop")
    except ValueError:
        # fallback: simple cut by min..max
        pop["part"] = pd.cut(pop["popularity"], bins=args.P, labels=labels, include_lowest=True)

    pop["part"] = pop["part"].astype("int16")
    # Join back to interactions by book_idx
    df = df.merge(pop[["book_idx", "part"]], on="book_idx", how="left")

    df.to_parquet(args.output, index=False)

    print(f"[RANGE-{args.col}] rows={len(df)}, P={args.P}")
    print(df["part"].value_counts().sort_index())

if __name__ == "__main__":
    main()
