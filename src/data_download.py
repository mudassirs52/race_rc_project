"""
data_download.py  —  Download & prepare the RACE dataset for Google Colab
Run this FIRST in Colab before any training.
"""
import os, subprocess, json, csv

def download_race():
    """Download RACE dataset from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        print("Downloading RACE dataset...")
        ds = load_dataset("race", "all")
        os.makedirs("data/raw", exist_ok=True)

        def to_csv(split, fname):
            rows = []
            for item in ds[split]:
                opts = item['options']
                rows.append({
                    'article': item['article'],
                    'question': item['question'],
                    'A': opts[0], 'B': opts[1],
                    'C': opts[2], 'D': opts[3],
                    'answer': item['answer'],
                })
            import pandas as pd
            pd.DataFrame(rows).to_csv(fname, index=False)
            print(f"  Saved {len(rows)} rows → {fname}")

        to_csv('train', 'data/raw/train.csv')
        to_csv('validation', 'data/raw/dev.csv')
        to_csv('test', 'data/raw/test.csv')
        print("✅ RACE dataset ready!")
    except Exception as e:
        print(f"Error: {e}\nTry: pip install datasets")

if __name__ == "__main__":
    download_race()
