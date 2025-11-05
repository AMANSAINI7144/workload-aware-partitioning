#!/usr/bin/env python3
import os, argparse, itertools, math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

def community_assign_round_robin(communities, sizes, P):
    """
    communities: list of sets of book_idx
    sizes: list of sizes (community sizes in interactions or books)
    Greedy: assign largest communities first to the currently smallest partition.
    """
    order = sorted(range(len(communities)), key=lambda i: sizes[i], reverse=True)
    part_sizes = [0]*P
    book_to_part = {}
    for idx in order:
        # pick partition with smallest current size
        p = int(np.argmin(part_sizes))
        for b in communities[idx]:
            book_to_part[b] = p
        part_sizes[p] += sizes[idx]
    return book_to_part, part_sizes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/interactions.parquet")
    ap.add_argument("--output", default="partitions/group_book_P4.parquet")
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--max_books_per_user", type=int, default=50,
                    help="cap number of books considered per user to bound pair explosion")
    ap.add_argument("--min_coaccess", type=int, default=2,
                    help="min times a pair co-occurs to keep an edge")
    ap.add_argument("--max_users", type=int, default=None,
                    help="optional cap on number of users (for speed if needed)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.read_parquet(args.input)[["user_idx","book_idx"]]

    if args.max_users is not None:
        # take most active users first (more signal)
        user_sizes = df.groupby("user_idx").size().sort_values(ascending=False)
        keep_users = set(user_sizes.head(args.max_users).index)
        df = df[df["user_idx"].isin(keep_users)]

    # build user -> list of book_idx (capped)
    user_groups = df.groupby("user_idx")["book_idx"].apply(list)

    # Co-access edge counting
    edge_counter = Counter()
    for books in tqdm(user_groups, desc="Co-access counting"):
        if len(books) > args.max_books_per_user:
            books = books[:args.max_books_per_user]
        # unique pairs only
        for i, j in itertools.combinations(sorted(set(books)), 2):
            edge_counter[(i, j)] += 1

    # Build graph with filtered edges
    G = nx.Graph()
    # add nodes present in data
    G.add_nodes_from(df["book_idx"].unique().tolist())
    kept = 0
    for (i, j), w in edge_counter.items():
        if w >= args.min_coaccess:
            G.add_edge(i, j, weight=int(w))
            kept += 1
    print(f"[GROUP] kept {kept} edges with weight >= {args.min_coaccess} "
          f"over {len(edge_counter)} total pairs")

    # If graph is very sparse, some nodes may be isolated: communities will be many singletons.
    # Label propagation communities:
    if G.number_of_edges() == 0:
        # degenerate: no co-access signal -> fallback to hash by book_idx
        print("[GROUP] No edges kept, falling back to hash(book_idx) assignment.")
        books = pd.Series(sorted(G.nodes()))
        parts = (books % args.P).astype("int16")
        book_to_part = dict(zip(books.tolist(), parts.tolist()))
    else:
        comms = list(nx.algorithms.community.label_propagation_communities(G))
        comms = [set(c) for c in comms]
        # size = number of books in community (can also weight by degree for better balance)
        sizes = [len(c) for c in comms]
        book_to_part, part_sizes = community_assign_round_robin(comms, sizes, args.P)
        print(f"[GROUP] communities={len(comms)}, part_sizes(books)={part_sizes}")

    # Map each interaction by its book's assigned partition
    df_full = pd.read_parquet(args.input)
    df_full["part"] = df_full["book_idx"].map(book_to_part).astype("Int16")
    # Unassigned (if any) -> simple hash(book_idx) % P
    missing = df_full["part"].isna().sum()
    if missing:
        hb = (pd.util.hash_pandas_object(df_full["book_idx"], index=False).astype("uint64") % args.P).astype("Int16")
        df_full["part"] = df_full["part"].fillna(hb)
        print(f"[GROUP] assigned {missing} missing via hash fallback")

    df_full["part"] = df_full["part"].astype("int16")
    df_full.to_parquet(args.output, index=False)

    print(f"[GROUP] rows={len(df_full)}, P={args.P}")
    print(df_full["part"].value_counts().sort_index())

if __name__ == "__main__":
    main()
