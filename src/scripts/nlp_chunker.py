# src/scripts/nlp_chunker.py

# routines to build dataset with different chunking strategies and
# pre-compute embeddings.

import argparse
import os
import pandas as pd
from pathlib import Path
import sys

sys.path.append("..")
from src.utils.nlp import split_into_sentences, split_into_noun_phrases
from src.utils.sysops import get_repo_root


def report_to_chunks(text, report_id, strategy) -> pd.DataFrame:
    # chunk a text and return index DataFrame with
    # index columns [report_id, sentence_id, [phrase_id], sentence, [phrase]]
    sentences = split_into_sentences(text)
    df = pd.DataFrame(
        {
            "report_id": [report_id] * len(sentences),
            "sentence_id": list(range(1, len(sentences) + 1)),  # 1-based rank
            "sentence": sentences,
        }
    )
    if strategy == "noph":  # split further into noun phrases
        noun_phrases = [split_into_noun_phrases(s) for s in sentences]
        phrase_ids = [[i + 1 for i in range(len(nps))] for nps in noun_phrases]
        df = df.loc[df.index.repeat([len(nps) for nps in noun_phrases])].reset_index(
            drop=True
        )
        df.insert(2, "phrase_id", [i for ids in phrase_ids for i in ids])
        df["phrase"] = [p for nps in noun_phrases for p in nps]
    return df


def reports_to_chunks(src_dir, tgt_dir, strategy):
    # calls report_to_chunk and writes back results
    # for each text file identified as a report in src_dir
    for report in os.listdir(src_dir):
        report_id = report.split(".")[0].split("_")[0]
        report_chunked = f"{report_id}.csv"
        src_file = os.path.join(src_dir, report)
        tgt_file = os.path.join(tgt_dir, report_chunked)
        if os.path.exists(tgt_file):
            skip = True
            try:
                tgt = pd.read_csv(tgt_file)
            except pd.errors.EmptyDataError:
                skip = False
            if tgt.shape[0] < 1:
                skip = False
            if skip:
                print(f"INFO\tSkipping {tgt_file} as already existing")
                continue

        print(f"STATUS\tReading {src_file} ...")
        with open(src_file, "r") as f1:
            text = f1.read()
            chunks_df = report_to_chunks(text, report_id, strategy)
            chunks_df.to_csv(tgt_file, index=False)
            print(f"STATUS:\tText chunks written to {tgt_file}")


parser = argparse.ArgumentParser(description="Chunk reports with specified strategy")
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
parser.add_argument(
    "--data_dir",
    type=str,
    required=False,
    default=Path(get_repo_root()) / "data",
    help="Data base dir",
)

args = parser.parse_args()


if __name__ == "__main__":

    src_dir = (
        Path(args.data_dir) / "02_processed" / args.benchmark / args.split / "full"
    )
    tgt_dir = (
        Path(args.data_dir)
        / "02_processed"
        / args.benchmark
        / args.split
        / args.strategy
    )
    reports_to_chunks(src_dir, tgt_dir, args.strategy)

# python scripts/nlp_chunker.py --split "test" --strategy "noph" --benchmark codiesp_en
# python scripts/nlp_chunker.py --split "train" --strategy "noph" --benchmark codiesp_en
# python scripts/nlp_chunker.py --split "test" --strategy "sentence" --benchmark codiesp_en
# python scripts/nlp_chunker.py --split "train" --strategy "sentence" --benchmark codiesp_en

## TODO: add routine for embeddings re-inference using te3l
## TODO: add routine for embeddings reduction via umap including model persistence using n_components 5, 10, 25, 50
