# src/scripts/icd10_embedder.py

# routines to build dataset with different chunking strategies and
# pre-compute embeddings.

import argparse
import numpy as np
import openai
import os
import pandas as pd
from pathlib import Path
import simple_icd_10_cm as icd10
import sys

sys.path.append("..")
from src.benchmarks.data import BenchmarkData
from src.utils.sysops import get_repo_root

BATCH_MAXSIZE = 2048


def infer_embeddings(descriptions, api_key):
    openai.api_key = api_key
    response = openai.embeddings.create(
        input=descriptions, model="text-embedding-3-large"
    )
    embeddings = [e.embedding for e in response.data]
    assert len(embeddings) == len(descriptions)
    print(f"INFO\tShape: {len(embeddings)} x {len(embeddings[0])}")
    return embeddings


def main(tgt_dir, api_key):
    print(f"INFO\tEmbeddings will be stored under {tgt_dir}")
    tgt_dir.mkdir(parents=True, exist_ok=True)
    # store index with code and description column and embeddings
    icd10_df = pd.DataFrame({"code": icd10.get_all_codes()})
    icd10_df["description"] = icd10_df["code"].apply(icd10.get_description)
    n = icd10_df.shape[0]

    for idx, start in enumerate(range(0, n, BATCH_MAXSIZE)):
        end = start + BATCH_MAXSIZE

        shard_id = f"{idx:02d}"
        npy_path = os.path.join(tgt_dir, f"icd10_shard{shard_id}.npy")
        csv_path = os.path.join(tgt_dir, f"icd10_shard{shard_id}.csv")

        if os.path.exists(npy_path):
            print(f"INFO\tSkip shard {shard_id} as already existing in {tgt_dir}")
            continue

        slice_i = icd10_df.iloc[start:end]
        slice_e = infer_embeddings(slice_i["description"], api_key)

        np.save(npy_path, slice_e)
        slice_i.to_csv(csv_path, index=False)

        print(f"INFO\tStored embeddings for {shard_id} under {tgt_dir}")


parser = argparse.ArgumentParser(description="Infer embeddings for chunked reports")

parser.add_argument("--api-key", type=str, required=True, help="API key for OpenAI")
parser.add_argument(
    "--data_dir",
    type=str,
    required=False,
    default=Path(get_repo_root()) / "data",
    help="Data base dir",
)


args = parser.parse_args()


if __name__ == "__main__":

    tgt_dir = (
        Path(args.data_dir) / "03_derived" / "icd10" / "te3l" / "leaf" / "original"
    )
    assert len(args.api_key)

    main(tgt_dir, api_key=args.api_key)


# Example call:
#   python scripts/icd10_embedder.py --api-key $OPENAI_API
