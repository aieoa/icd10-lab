# src/scripts/umap_reducer.py

# routines to build dataset with different chunking strategies and
# pre-compute embeddings.

import argparse
import joblib
import numpy as np
from pathlib import Path
import sys
import umap


sys.path.append("..")
from src.utils.sysops import get_repo_root
import src.scripts.umap_runner as umap_runner


def write_back_split(e_umap, src_files, file_lengths, tgt_path, file_count: int):
    # e_train_umap, e_train_files, e_train_lengths, tgt_dirs['train'])
    tgt_path.mkdir(parents=True, exist_ok=True)
    print(f"INFO\tReduced embeddings will be written to {tgt_path}")
    start = 0
    assert file_count == len(file_lengths)
    for f, length in zip(src_files, file_lengths):
        report_id = f.stem.split("_")[0]
        end = start + length
        # Sliced result for this source file
        slice = e_umap[start:end]
        tgt_file = tgt_path / f"{report_id}.npy"
        np.save(tgt_file, slice)
        start = end
        print(f"INFO\tReduced embeddings written to {tgt_file}")


def main(src_dirs, tgt_dirs, strategy, benchmark, seed, n_components):
    benchmark_file_ctrs = {"codiesp_en": {"train": 500, "test": 250}}
    print(src_dirs["train"])
    e_train_files = sorted(src_dirs["train"].glob("*.npy"))
    e_train_list = [np.load(f) for f in e_train_files]
    e_train_lengths = [np.load(f).shape[0] for f in e_train_files]
    print(len(e_train_list))
    assert len(e_train_list) == benchmark_file_ctrs[benchmark]["train"]

    e_train = np.concatenate(e_train_list, axis=0)

    e_test_files = sorted(src_dirs["test"].glob("*.npy"))
    e_test_list = [np.load(f) for f in e_test_files]
    e_test_lengths = [np.load(f).shape[0] for f in e_test_files]
    assert len(e_test_list) == benchmark_file_ctrs[benchmark]["test"]
    e_test = np.concatenate(e_test_list, axis=0)

    umap_dir = Path(get_repo_root()) / "models"
    model_path = umap_dir / f"umap_model_{benchmark}_{strategy}_{n_components}.joblib"
    umap_dir.mkdir(parents=True, exist_ok=True)
    print(f"INFO\tModel will be written to {model_path}")
    umap_transformer = umap.UMAP(n_components=n_components, random_state=seed)

    # Transform training data
    print("STATUS\tFitting and transforming training data...")
    e_train_umap = umap_transformer.fit_transform(e_train)

    # Save the fitted model
    joblib.dump(umap_transformer, model_path)
    print(f"INFO\tSaved UMAP model to {model_path}")

    print("STATUS\tTransforming test data...")
    e_test_umap = umap_transformer.transform(e_test)

    # Split and write-back report-wise
    write_back_split(
        e_test_umap,
        e_test_files,
        e_test_lengths,
        tgt_dirs["test"],
        benchmark_file_ctrs[benchmark]["test"],
    )
    write_back_split(
        e_train_umap,
        e_train_files,
        e_train_lengths,
        tgt_dirs["train"],
        benchmark_file_ctrs[benchmark]["train"],
    )


parser = argparse.ArgumentParser(
    description="Reduce embeddings for train and test splits"
)
parser.add_argument(
    "--benchmark",
    type=str,
    required=False,
    default="codiesp_en",
    help="Benchmark dataset to chunk (eg codiesp_en, mimiciv, charite)",
)

parser.add_argument(
    "--strategy", type=str, required=True, help="Chunking strategy name (eg noph, sent)"
)

parser.add_argument(
    "--n-components", type=int, required=True, help="Number of target dimensions"
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
    assert args.strategy in [
        "noph",
        "sent",
    ], "supported chunking strategies: [noph, sent]"
    base_dir = Path(args.data_dir) / "03_derived" / args.benchmark
    src_dirs = {
        "train": base_dir / "train" / "te3l" / args.strategy / "original",
        "test": base_dir / "test" / "te3l" / args.strategy / "original",
    }
    tgt_dirs = {
        "train": base_dir
        / "train"
        / "te3l"
        / args.strategy
        / "reduced"
        / f"umap_{args.n_components}",
        "test": base_dir
        / "test"
        / "te3l"
        / args.strategy
        / "reduced"
        / f"umap_{args.n_components}",
    }

    main(
        src_dirs,
        tgt_dirs,
        strategy=args.strategy,
        benchmark=args.benchmark,
        n_components=args.n_components,
        seed=args.seed,
    )

# Example call:
# python scripts/umap_reducer.py --benchmark codiesp_en --strategy noph --n-components 10
