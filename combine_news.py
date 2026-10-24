"""Combine ISOT-style True.csv and Fake.csv into data/news.csv.

Usage:
  python combine_news.py --real data/True.csv --fake data/Fake.csv
  python combine_news.py --real True.csv --fake Fake.csv --output data/news.csv

The script labels rows from True.csv as REAL and Fake.csv as FAKE. Only use
files whose source/label semantics you have verified.
"""
import argparse
from pathlib import Path
import pandas as pd

TEXT_CANDIDATES = ("text", "article", "content", "body", "news", "description")

def find_text_column(df: pd.DataFrame) -> str:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in TEXT_CANDIDATES:
        if name in lower:
            return lower[name]
    raise ValueError(f"Could not identify article text column. Available columns: {list(df.columns)}. Use --text-col explicitly.")

def load_one(path: Path, label: str, text_col: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path.resolve()}")
    frame = pd.read_csv(path, low_memory=False)
    column = text_col or find_text_column(frame)
    if column not in frame.columns:
        raise ValueError(f"Text column {column!r} not found in {path.name}. Available: {list(frame.columns)}")
    out = pd.DataFrame({"text": frame[column].fillna("").astype(str).str.strip(), "label": label})
    out = out[out["text"].str.len() >= 20]
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", default="data/True.csv", help="CSV containing verified REAL articles")
    parser.add_argument("--fake", default="data/Fake.csv", help="CSV containing verified FAKE articles")
    parser.add_argument("--output", default="data/news.csv", help="Output combined labelled CSV")
    parser.add_argument("--real-text-col", default=None, help="Optional text column name in real CSV")
    parser.add_argument("--fake-text-col", default=None, help="Optional text column name in fake CSV")
    args = parser.parse_args()
    real = load_one(Path(args.real), "REAL", args.real_text_col)
    fake = load_one(Path(args.fake), "FAKE", args.fake_text_col)
    combined = pd.concat([real, fake], ignore_index=True)
    combined["text_norm"] = combined["text"].str.lower().str.replace(r"\\s+", " ", regex=True).str.strip()
    combined = combined.drop_duplicates("text_norm").drop(columns="text_norm").reset_index(drop=True)
    if combined["label"].nunique() != 2:
        raise ValueError("Combined dataset must contain both REAL and FAKE labels.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, encoding="utf-8")
    print(f"Saved: {output.resolve()}")
    print(f"Rows: {len(combined):,}")
    print("Label counts:")
    print(combined["label"].value_counts().to_string())
    print("Columns: text, label")

if __name__ == "__main__":
    main()
