import os
import pandas as pd

INPUT_DIR  = "./nba_data"
OUTPUT_DIR = "./nba_data_cleaned"

SEASON_COLS = ["SEASON", "SEASON_ID", "SEASON_YEAR"]

EXCLUDE_SEASONS = {"2025-26", "2025-2026", "22025", "2025"}


def drop_2025_26(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    original_len = len(df)
    for col in SEASON_COLS:
        if col not in df.columns:
            continue
        mask = df[col].astype(str).str.strip().isin(EXCLUDE_SEASONS)
        mask |= df[col].astype(str).str.startswith("22025")
        df = df[~mask]
    return df, original_len - len(df)


def fill_empty_with_na(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = df.isna().sum().sum()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].where(
            df[col].astype(str).str.strip() != "",
            other=pd.NA
        )

    df = df.where(df.notna(), other=pd.NA)

    after = df.isna().sum().sum()
    return df, int(after - before)


def clean_file(src_path: str, dst_path: str) -> None:
    fname = os.path.basename(src_path)
    print(f"\n  Processing: {fname}")

    try:
        df = pd.read_csv(src_path, low_memory=False)
        print(f"    Loaded  : {len(df):>10,} rows  ×  {len(df.columns)} cols")
    except Exception as e:
        print(f"    ✗ Could not read file: {e}")
        return

    df, dropped = drop_2025_26(df)
    if dropped:
        print(f"    Dropped : {dropped:>10,} rows  (2025-26 season)")
    else:
        print(f"    Dropped :          0 rows  (no 2025-26 data found)")

    df, filled = fill_empty_with_na(df)
    print(f"    Filled  : {filled:>10,} empty cells  → N/A")

    try:
        df.to_csv(dst_path, index=False, na_rep="N/A")
        mb = os.path.getsize(dst_path) / 1_000_000
        print(f"    Saved   : {len(df):>10,} rows  →  {dst_path}  ({mb:.1f} MB)")
    except Exception as e:
        print(f"    ✗ Could not save file: {e}")


def main():
    print("=" * 62)
    print("  🏀  NBA Data Cleaner")
    print("=" * 62)
    print(f"  Input  : {os.path.abspath(INPUT_DIR)}")
    print(f"  Output : {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Rules  :")
    print(f"    • Drop all 2025-26 season rows")
    print(f"    • Empty cells → N/A  (zeros are untouched)")
    print("=" * 62)

    if not os.path.isdir(INPUT_DIR):
        print(f"\n  ✗ Input directory '{INPUT_DIR}' not found.")
        print(f"    Make sure you run this from the same folder as the scraper.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".csv") and not f.startswith("_ckpt_")
    )

    if not csv_files:
        print(f"\n  ✗ No CSV files found in '{INPUT_DIR}'.")
        return

    print(f"\n  Found {len(csv_files)} file(s) to clean:\n")
    for f in csv_files:
        print(f"    {f}")

    for fname in csv_files:
        src = os.path.join(INPUT_DIR, fname)
        dst = os.path.join(OUTPUT_DIR, fname)
        clean_file(src, dst)

    print("\n" + "=" * 62)
    print("  ✅  All done — cleaned files are in:", os.path.abspath(OUTPUT_DIR))
    print("=" * 62)


if __name__ == "__main__":
    main()