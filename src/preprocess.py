import argparse
import os
import sys
import json
import textwrap
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import yaml
except ImportError:
    yaml = None


# ---------- Defaults ----------
DEFAULT_INPUT = "data/Books_rating.csv"
DEFAULT_OUT_PARQUET = "data/interactions.parquet"
DEFAULT_SUMMARY = "results/preprocess_summary.txt"
DEFAULT_MIN_INTERACTIONS = 5
DEFAULT_SEED = 42


def load_config(config_path: str | None):
    cfg = {}
    if config_path and os.path.exists(config_path):
        if yaml is None:
            print("[WARN] pyyaml not installed; ignoring config.yml", file=sys.stderr)
        else:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
    return cfg


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to map dataset's original column names to canonical ones.
    Expected originals (from your sample):
      - 'User_id' -> user_id
      - 'Id'      -> book_id
      - 'review/score' -> rating
      - 'review/time'  -> timestamp
    We keep extra columns untouched.
    """
    rename_map = {}
    cols = {c.lower(): c for c in df.columns}  # case-insensitive lookup

    # helper to map if present (case-insensitively)
    def map_col(src_lower, dest):
        if src_lower in cols:
            rename_map[cols[src_lower]] = dest

    map_col("user_id", "user_id")           # just in case it already is user_id
    map_col("user_id", "user_id")           # redundant but safe
    map_col("user_id", "user_id")
    map_col("user_id", "user_id")

    # sample file has 'User_id'
    map_col("user_id", "user_id")
    if "user_id" not in rename_map.values():
        map_col("user_id", "user_id")  # fallback

    # explicit mappings based on your CSV headers
    map_col("user_id", "user_id")
    if "User_id" in df.columns:
        rename_map["User_id"] = "user_id"
    if "Id" in df.columns:
        rename_map["Id"] = "book_id"
    if "review/score" in df.columns:
        rename_map["review/score"] = "rating"
    if "review/time" in df.columns:
        rename_map["review/time"] = "timestamp"

    # Final rename
    df = df.rename(columns=rename_map)

    # Keep only needed columns if they exist
    needed = ["user_id", "book_id", "rating", "timestamp"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after renaming: {missing}\n"
            f"Columns present: {list(df.columns)}"
        )
    return df[needed]


def coerce_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace on IDs
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["book_id"] = df["book_id"].astype(str).str.strip()

    # rating -> float; timestamp -> int
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")

    # Drop rows with NA in required fields
    before = len(df)
    df = df.dropna(subset=["user_id", "book_id", "rating", "timestamp"])
    df["timestamp"] = df["timestamp"].astype(np.int64, errors="ignore")
    after = len(df)
    print(f"[INFO] Dropped {before - after} rows with missing/invalid values")

    # Deduplicate exact rows
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "book_id", "rating", "timestamp"])
    print(f"[INFO] Dropped {before - len(df)} exact duplicate rows")

    return df


def dedup_latest_per_user_book(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the latest rating for each (user, book)
    df = df.sort_values(["user_id", "book_id", "timestamp"])
    latest = df.groupby(["user_id", "book_id"], as_index=False).tail(1)
    print(f"[INFO] After keeping latest per (user, book): {len(latest)} rows")
    return latest


def map_to_indices(df: pd.DataFrame) -> pd.DataFrame:
    user_codes = pd.Categorical(df["user_id"])
    book_codes = pd.Categorical(df["book_id"])
    df["user_idx"] = user_codes.codes.astype(np.int32)
    df["book_idx"] = book_codes.codes.astype(np.int32)
    print(f"[INFO] Unique users: {user_codes.categories.size}, books: {book_codes.categories.size}")
    return df


def filter_infrequent(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    # Filter users
    user_cnt = df.groupby("user_idx").size()
    keep_users = set(user_cnt[user_cnt >= min_interactions].index.tolist())
    df = df[df["user_idx"].isin(keep_users)]

    # Recompute books after user filter
    book_cnt = df.groupby("book_idx").size()
    keep_books = set(book_cnt[book_cnt >= min_interactions].index.tolist())
    df = df[df["book_idx"].isin(keep_books)]

    print(
        f"[INFO] After filtering (min_interactions={min_interactions}): "
        f"{len(df)} rows, {df['user_idx'].nunique()} users, {df['book_idx'].nunique()} books"
    )
    return df


def compute_popularity(df: pd.DataFrame) -> pd.DataFrame:
    pop = df.groupby("book_idx").size().rename("popularity")
    df = df.join(pop, on="book_idx")
    return df


def write_summary(df: pd.DataFrame, path: str, seed: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append(f"Preprocess summary - generated: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Random seed: {seed}")
    lines.append("")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Unique users: {df['user_idx'].nunique()}")
    lines.append(f"Unique books: {df['book_idx'].nunique()}")
    lines.append("")
    lines.append("Rating stats:")
    lines.append(str(df["rating"].describe()))
    lines.append("")
    lines.append("Popularity stats (per interaction row):")
    lines.append(str(df["popularity"].describe()))
    lines.append("")
    lines.append("Head:")
    lines.append(df.head(5).to_string(index=False))

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Wrote summary -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Amazon Books ratings into clean parquet for partitioning.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to Books_rating.csv")
    parser.add_argument("--output", default=DEFAULT_OUT_PARQUET, help="Output Parquet path")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY, help="Summary txt path")
    parser.add_argument("--min_interactions", type=int, default=DEFAULT_MIN_INTERACTIONS,
                        help="Min interactions per user and per book to keep")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--config", default="src/config.yml", help="Optional YAML config")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle final rows before save")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", DEFAULT_SEED)
    np.random.seed(seed)

    # Allow min_interactions override via config.yml
    min_int = cfg.get("filters", {}).get("min_user_interactions", args.min_interactions)
    # Use same threshold for books
    min_book = cfg.get("filters", {}).get("min_book_interactions", min_int)

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading: {args.input}")
    df_raw = pd.read_csv(args.input)

    print("[INFO] Selecting & renaming columns…")
    df = normalize_columns(df_raw)

    print("[INFO] Coercing types & cleaning…")
    df = coerce_and_clean(df)

    print("[INFO] Keeping latest review per (user, book)…")
    df = dedup_latest_per_user_book(df)

    print("[INFO] Mapping IDs to dense integer indices…")
    df = map_to_indices(df)

    print(f"[INFO] Filtering users/books with >= {min_int} interactions…")
    df = filter_infrequent(df, min_interactions=min_int)

    print("[INFO] Computing popularity (ratings per book)…")
    df = compute_popularity(df)

    if args.shuffle:
        print("[INFO] Shuffling rows…")
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Ensure output dirs
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)

    print(f"[INFO] Saving parquet -> {args.output}")
    df.to_parquet(args.output, index=False)

    print("[INFO] Writing summary…")
    write_summary(df, args.summary, seed)

    print("[DONE] Preprocessing complete.")


if __name__ == "__main__":
    main()