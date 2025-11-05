#!/usr/bin/env python3
import os, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/interactions.parquet")
    ap.add_argument("--output", default="partitions/hash_user_P4.parquet")
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--key", default="user_idx", help="column to hash")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.read_parquet(args.input)

    if args.key not in df.columns:
        raise SystemExit(f"Key column '{args.key}' not found. Available: {list(df.columns)}")

    # stable 64-bit hash → modulo P
    h = pd.util.hash_pandas_object(df[args.key], index=False).astype("uint64")
    df["part"] = (h % args.P).astype("int16")

    df.to_parquet(args.output, index=False)

    print(f"[HASH-{args.key}] rows={len(df)}, P={args.P}")
    print(df["part"].value_counts().sort_index())

if __name__ == "__main__":
    main()
