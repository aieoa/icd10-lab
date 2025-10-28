# src/scripts/icd10_umap_reducer.py

# reduce original embeddings via given umap model

import argparse
import joblib
import numpy as np
from pathlib import Path
import re
import sys


sys.path.append("..")
from src.utils.sysops import get_repo_root


def main(src_dir, tgt_dir, model_path, seed):
    print(f"STATUS\tLoading {model_path.stem}")
    umap_transformer = joblib.load(model_path)

    print(f"INFO\tOriginal embeddings are read from {src_dir}")
    print(f"INFO\tReduced embeddings will be stored under {tgt_dir}")
    tgt_dir.mkdir(parents=True, exist_ok=True)
    shard_files = [f for f in src_dir.glob("*.npy")]
    for i, shard_file in enumerate(shard_files):
        print(f"STATUS\tProcessing shard {i+1} / {len(shard_files)}")
        e = np.load(shard_file)
        e_umap = umap_transformer.transform(e)
        shard_file_umap = tgt_dir / shard_file.name
        np.save(shard_file_umap, e_umap)


parser = argparse.ArgumentParser(description="Infer embeddings for chunked reports")
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
    help="Umap model path for dimensionality reduction",
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=False,
    default=Path(get_repo_root()) / "data",
    help="Data base dir",
)
parser.add_argument(
    "--seed",
    type=int,
    required=False,
    default=81,
    help="Seed for UMAP algorithm",
)

args = parser.parse_args()


if __name__ == "__main__":

    model_path = Path(args.model_path)
    assert model_path.is_file()
    match = re.search(r"(\d+)(?!.*\d)", args.model_path)
    assert match
    n_components = match.group(1)
    strategy = None
    if "_sent_" in model_path.name:
        strategy = "sent"
    elif "_noph_" in model_path.name:
        strategy = "noph"
    assert strategy

    src_dir = (
        Path(args.data_dir) / "03_derived" / "icd10" / "te3l" / "leaf" / "original"
    )

    tgt_dir = (
        Path(args.data_dir)
        / "03_derived"
        / "icd10"
        / "te3l"
        / "leaf"
        / "reduced"
        / f"umap_{strategy}_{n_components}"
    )

    assert tgt_dir != src_dir
    main(src_dir, tgt_dir, model_path, args.seed)


# Example call:
#   python scripts/icd10_umap_reducer.py --model-path ../models/umap_model_codiesp_en_sent_50.joblib
