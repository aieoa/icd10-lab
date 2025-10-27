# src/scripts/chunk_embedder.py

# routines to build dataset with different chunking strategies and
# pre-compute embeddings.

import argparse
import numpy as np
import openai
import os
import pandas as pd
from pathlib import Path
import sys

sys.path.append("..")
from src.utils.sysops import get_repo_root


def infer_embeddings(chunk_list, api_key):

    openai.api_key = api_key

    # Batch query for embeddings
    response = openai.embeddings.create(
        input=chunk_list, model="text-embedding-3-large"
    )
    embeddings = [e.embedding for e in response.data]
    assert len(embeddings) == len(chunk_list)
    print(f"INFO\tShape: {len(embeddings)} x {len(embeddings[0])}")
    return embeddings


def main(src_dir, tgt_dir, strategy, api_key):
    src_column = {"noph": "phrase", "sent": "sentence"}
    tgt_dir = Path(tgt_dir)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    print(f"INFO\tEmbeddings written in npy format to {tgt_dir}")

    for report in src_dir.glob("*.csv"):
        print(f"STATUS\tReading {report} ...")
        report_id = report.stem.split("_")[0]
        tgt_file = tgt_dir / f"{report_id}.npy"
        if os.path.exists(tgt_file) and os.path.getsize(tgt_file) > 0:
            print(f"INFO\tSkipping {tgt_file} as already existing")
            continue

        chunk_series = pd.read_csv(report, usecols=[src_column[strategy]])
        chunk_list = chunk_series[src_column[strategy]].astype(str).fillna("").tolist()

        e_list = infer_embeddings(chunk_list, api_key)
        e_array = np.array(e_list, dtype=np.float32)
        assert e_array.size > len(e_list) * 10
        np.save(tgt_file, e_array)
        print(f"STATUS:\tChunk embeddings written to {tgt_file}")


parser = argparse.ArgumentParser(description="Infer embeddings for chunked reports")
parser.add_argument(
    "--benchmark",
    type=str,
    required=False,
    default="codiesp_en",
    help="Benchmark dataset to chunk (eg codiesp_en, mimiciv, charite)",
)

parser.add_argument(
    "--split", type=str, required=True, help="Data split type (eg train, test, val)"
)
parser.add_argument(
    "--strategy", type=str, required=True, help="Chunking strategy name (eg noph, sent)"
)
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
    assert args.strategy in [
        "noph",
        "sent",
    ], "supported chunking strategies: [noph, sent]"
    src_dir = (
        Path(args.data_dir)
        / "02_processed"
        / args.benchmark
        / args.split
        / args.strategy
    )
    assert src_dir.exists()

    tgt_dir = (
        Path(args.data_dir)
        / "03_derived"
        / args.benchmark
        / args.split
        / "te3l"
        / args.strategy
        / "original"
    )
    assert len(args.api_key)
    main(src_dir, tgt_dir, args.strategy, api_key=args.api_key)


# Example call:
#   python chunk_embedder.py --benchmark codiesp_en --split test --strategy sent --api-key $OPENAI_API
