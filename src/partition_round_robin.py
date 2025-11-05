#!/usr/bin/env python3
import os, argparse
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/interactions.parquet")
    ap.add_argument("--output", default="partitions/round_robin_P4.parquet")
    ap.add_argument("--P", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.read_parquet(args.input)
    n = len(df)
    df = df.reset_index(drop=True)
    df["part"] = np.arange(n, dtype=np.int64) % args.P

    df.to_parquet(args.output, index=False)

    print(f"[RR] rows={n}, P={args.P}")
    print(df["part"].value_counts().sort_index())

if __name__ == "__main__":
    main()
    