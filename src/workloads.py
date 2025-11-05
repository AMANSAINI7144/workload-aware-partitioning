#!/usr/bin/env python3
"""
Generate reproducible workload specs for Option C.
Outputs JSON files in ./workloads/:
  - W1_user_lookup.json
  - W2_range_popularity.json
  - W3_book_bundles.json
"""

import os, json, argparse, numpy as np, pandas as pd
from collections import Counter
from itertools import combinations

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"[OK] wrote {path}")

def build_W1(df, k_users=500, min_interactions=5, seed=42):
    np.random.seed(seed)
    # users with at least min_interactions
    user_sizes = df.groupby("user_idx").size()
    cand = user_sizes[user_sizes >= min_interactions].index.to_numpy()
    if len(cand) == 0:
        raise SystemExit("No users meet min_interactions for W1.")
    if k_users > len(cand): k_users = len(cand)
    users = np.random.choice(cand, size=k_users, replace=False).tolist()
    return {"type": "W1_user_lookup", "k": k_users, "min_interactions": min_interactions, "users": users}

def build_W2(df, k_ranges=300, window_quantile=0.10, seed=42):
    """
    Range queries over popularity. We pick random centers in popularity quantiles
    and create windows of +/- (window_quantile/2) of the quantile span.
    """
    np.random.seed(seed)
    if "popularity" not in df.columns:
        raise SystemExit("Column 'popularity' missing. Run preprocessing first.")
    # Use popularity per interaction row; get quantiles to define window edges
    q = df["popularity"].quantile(np.linspace(0, 1, 101))  # 0..100%
    ranges = []
    for _ in range(k_ranges):
        c = np.random.uniform(0.05, 0.95)  # avoid extremes
        half = window_quantile / 2.0
        loq, hiq = max(0.0, c - half), min(1.0, c + half)
        lo = float(df["popularity"].quantile(loq))
        hi = float(df["popularity"].quantile(hiq))
        if hi < lo: lo, hi = hi, lo
        ranges.append({"lo": lo, "hi": hi})
    return {"type": "W2_range_popularity", "k": k_ranges, "window_quantile": window_quantile, "ranges": ranges}

def build_W3(df, max_users=30000, min_pair_coaccess=3, bundle_size=3, k_bundles=300, seed=42):
    """
    Build book bundles based on co-access. To keep fast, cap users considered.
    We compute top co-access pairs, then expand to bundles of size 'bundle_size'
    by merging frequent neighbors around a seed book.
    """
    np.random.seed(seed)
    # optionally restrict to most active users for speed/signal
    user_sizes = df.groupby("user_idx").size().sort_values(ascending=False)
    if max_users is not None:
        keep_users = set(user_sizes.head(max_users).index)
        dfx = df[df["user_idx"].isin(keep_users)]
    else:
        dfx = df

    # build user -> unique books
    user_books = dfx.groupby("user_idx")["book_idx"].apply(lambda s: sorted(set(s.tolist())))
    # count co-access pairs
    pair_counter = Counter()
    for books in user_books:
        # limit very large histories to reduce explosion
        if len(books) > 80:
            books = books[:80]
        for i, j in combinations(books, 2):
            pair_counter[(i, j)] += 1

    # keep only frequent pairs
    frequent_pairs = [(ij, c) for ij, c in pair_counter.items() if c >= min_pair_coaccess]
    if not frequent_pairs:
        # fallback: pick top popular books and make random bundles
        popular_books = df.groupby("book_idx").size().sort_values(ascending=False).head(2000).index.to_numpy()
        bundles = []
        for _ in range(k_bundles):
            bundles.append(list(np.random.choice(popular_books, size=bundle_size, replace=False)))
        return {"type": "W3_book_bundles", "k": k_bundles, "min_pair_coaccess": min_pair_coaccess,
                "bundle_size": bundle_size, "bundles": [{"book_idxs": b} for b in bundles]}

    # adjacency for simple bundle grow
    adj = {}
    for (i, j), c in frequent_pairs:
        adj.setdefault(i, []).append((j, c))
        adj.setdefault(j, []).append((i, c))
    # sort neighbors by weight desc
    for v in adj:
        adj[v].sort(key=lambda x: -x[1])

    # seed from top-degree books
    degrees = {v: sum(c for _, c in nbrs) for v, nbrs in adj.items()}
    seeds = [v for v, _ in sorted(degrees.items(), key=lambda x: -x[1])]
    if len(seeds) > 5000:
        seeds = seeds[:5000]

    bundles = []
    used = set()
    for s in seeds:
        if len(bundles) >= k_bundles: break
        if s in used: continue
        bundle = [s]
        used.add(s)
        for nb, _w in adj.get(s, []):
            if nb not in used:
                bundle.append(nb); used.add(nb)
                if len(bundle) >= bundle_size: break
        if len(bundle) == bundle_size:
            bundles.append(bundle)

    # if still short, fill randomly from frequent books
    if len(bundles) < k_bundles:
        frequent_books = np.array(list(degrees.keys()))
        while len(bundles) < k_bundles and len(frequent_books) >= bundle_size:
            bset = np.random.choice(frequent_books, size=bundle_size, replace=False).tolist()
            bundles.append(bset)

    return {"type": "W3_book_bundles", "k": k_bundles, "min_pair_coaccess": min_pair_coaccess,
            "bundle_size": bundle_size, "bundles": [{"book_idxs": b} for b in bundles]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/interactions.parquet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w1_users", type=int, default=500)
    ap.add_argument("--w2_ranges", type=int, default=300)
    ap.add_argument("--w3_bundles", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)[["user_idx","book_idx","popularity"]]

    w1 = build_W1(df, k_users=args.w1_users, min_interactions=5, seed=args.seed)
    w2 = build_W2(df, k_ranges=args.w2_ranges, window_quantile=0.10, seed=args.seed)
    w3 = build_W3(df, max_users=30000, min_pair_coaccess=3, bundle_size=3, k_bundles=args.w3_bundles, seed=args.seed)

    save_json("workloads/W1_user_lookup.json", w1)
    save_json("workloads/W2_range_popularity.json", w2)
    save_json("workloads/W3_book_bundles.json", w3)

if __name__ == "__main__":
    main()
