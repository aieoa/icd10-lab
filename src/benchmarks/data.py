import os
import torch
import pandas as pd
from pathlib import Path
import simple_icd_10_cm as icd
from typing import List, Dict, Any
import logging

import src.utils.sysops as sysops

logger = logging.getLogger(__name__)


class BenchmarkData:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = cfg.get("device", None)
        self.benchmark_str = cfg["benchmark"].lower()
        self.data_dir = cfg["data_dir"]  # base data directory
        self.chunking_strategy = cfg["chunking_strategy"]
        self.llm = cfg["embedding_model"]
        self.reducer = cfg.get("reducer", None)
        self.reducer_params = cfg.get("reducer_params", {})
        self.icd10_params = cfg["icd10_params"]
        self.terminals_only = self.icd10_params.get("terminals_only", True)
        self.subset = {
            "test": cfg["subset"].get("test", []),
            "train": cfg["subset"].get("train", []),
        }

        if self.reducer is None:
            icd10_e_fmt = "last_layer_{:d}.pt"
            icd10_i_fmt = "last_layer_{:d}.pt.idx"
        else:
            icd10_e_fmt = f"last_layer_all.pt"
            icd10_i_fmt = f"last_layer_all.pt.idx"
        self._icd10_e_fmt = icd10_e_fmt
        self._icd10_i_fmt = icd10_i_fmt

        self._setup_paths()
        self._set_device()
        # Load indices/labels for train/test/reference
        self._load_all_indices()
        # Load embeddings for train/test/reference
        self._load_all_embeddings()
        # Load reference embeddings and codes
        self._load_icd10(num_shards=10, terminals_only=self.terminals_only)

    def _set_device(self):
        if self.device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        logger.info(f"Using device: {device}")
        self.device = device

    def _setup_paths(self):
        # Setup paths for embeddings, indices, and reference
        if not os.path.isabs(self.data_dir):
            if self.data_dir.startswith("~"):
                self.data_dir = os.path.expanduser(self.data_dir)
            else:
                repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.data_dir = os.path.join(repo_dir, self.data_dir)
        if self.reducer:
            subdir = (
                Path("reduced")
                / f"{self.reducer}_{self.reducer_params['n_components']}"
            )
        else:
            subdir = "original"
        benchmark_subdir = os.path.join(
            self.cfg["embedding_model"], self.cfg["chunking_strategy"], subdir
        )

        self.paths = {
            "test_e": Path(self.data_dir)
            / "03_derived"
            / self.benchmark_str
            / "test"
            / benchmark_subdir,
            "train_e": Path(self.data_dir)
            / "03_derived"
            / self.benchmark_str
            / "train"
            / benchmark_subdir,
            "icd10_e": Path(self.data_dir)
            / "03_derived"
            / "icd10"
            / self.cfg["embedding_model"]
            / self.cfg["icd10_params"]["aggregate_method"]
            / subdir,
            "test_i": Path(self.data_dir)
            / "02_processed"
            / self.benchmark_str
            / "test"
            / self.cfg["chunking_strategy"],
            "train_i": Path(self.data_dir)
            / "02_processed"
            / self.benchmark_str
            / "train"
            / self.cfg["chunking_strategy"],
            "icd10_i": Path(self.data_dir)
            / "03_derived"
            / "icd10"
            / self.cfg["embedding_model"]
            / self.cfg["icd10_params"]["aggregate_method"]
            / subdir,
        }
        for path in self.paths.values():
            assert os.path.exists(
                path
            ), f"Path does not exist: {path}, correct or precompute embeddings."

    def _load_all_indices(self):
        # Load indices for train/test/reference
        self.train_i = self._load_indices("train", subset=self.subset["train"])
        self.test_i = self._load_indices("test", subset=self.subset["test"])

    def _load_all_embeddings(self):
        # Load embeddings for train/test/reference
        self.train_e = self._load_embeddings("train")
        self.test_e = self._load_embeddings("test")

    def _load_indices(self, split: str, subset: List = None):
        # todo: load here dvc versioned data set with associated index
        dataset_dir = self.paths[f"{split}_i"]
        if self.cfg["chunking_strategy"] == "full":  # txt files
            index_files = [f for f in os.listdir(dataset_dir) if f.endswith(".txt")]
        else:  # csv files
            index_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
        indices = pd.DataFrame()
        for fname in index_files:
            report_id = fname.split(".")[0].split("_")[0]
            if report_id == "S1139-76322014000500011-1":
                continue
            if len(subset) and report_id not in subset:
                continue
            i = pd.read_csv(os.path.join(dataset_dir, fname))
            if "report_id" not in i.columns:
                i["report_id"] = report_id
            indices = pd.concat([indices, i], axis=0, ignore_index=True)
        return None if indices.empty else indices

    def _load_embeddings(self, split: str):
        # TODO: repair corrupt file report_id = 'S1139-76322014000500011-1'
        # TODO: figure out if embeddings are torch tensors or numpy arrays in advance and set weights_only accordingly (False for numpy arrays, True for torch tensors)
        weights_only = self.reducer is None

        # Load embeddings and index DataFrame for a split
        embedding_dir = self.paths[f"{split}_e"]
        embeddings = []
        for fname in os.listdir(embedding_dir):
            if fname.endswith(".pt"):
                report_id = fname.replace("_en.pt", "")
                if report_id == "S1139-76322014000500011-1":
                    continue
                pt_path = os.path.join(embedding_dir, fname)
                e = torch.load(
                    pt_path, map_location=self.device, weights_only=weights_only
                )
                embeddings.extend(e)
        try:
            embeddings = torch.stack([torch.tensor(e) for e in embeddings], dim=0)
        except Exception:
            embeddings = torch.stack(embeddings, dim=0)
        assert len(embeddings)
        return embeddings if len(embeddings) else None

    def _load_icd10_one_shard(self, icd10_dir, terminals_only: bool = True):
        # Load reference embeddings and codes from a single file

        logger.info(f"Loading reference embeddings from {icd10_dir}")
        pt_file = os.path.join(icd10_dir, "last_layer_all.pt")
        idx_file = os.path.join(icd10_dir, "last_layer_all.pt.idx")
        icd10_e = sysops.load_tensor(pt_file, device=self.device)
        with open(idx_file, "r") as f:
            icd10_i = [line.strip() for line in f]
        # Reduce to terminal codes if specified
        if terminals_only:
            ei = [
                (u, v)
                for u, v in zip(icd10_e, icd10_i)
                if icd.is_valid_item(v) and icd.is_leaf(v)
            ]
            print(f"INFO\treduced to {len(ei)} terminals")
        else:
            ei = [(u, v) for u, v in zip(icd10_e, icd10_i) if icd.is_valid_item(v)]
        icd10_e = [torch.tensor(u) for u, v in ei]
        icd10_i = [v for u, v in ei]
        self.icd10_e = icd10_e
        self.icd10_i = icd10_i

    def _load_icd10_multiple_shards(
        self, icd10_dir, num_shards: int = 10, terminals_only: bool = True
    ):
        def read_codes(file_name):
            with open(file_name, "r") as file:
                for code in file:
                    yield code.strip()

        icd10_e = []
        icd10_i = []
        for i in range(num_shards):
            pt_file = icd10_dir / self._icd10_e_fmt.format(i + 1)
            idx_file = icd10_dir / self._icd10_i_fmt.format(i + 1)
            logger.info(f"Loading reference embeddings from {pt_file}")
            e = torch.load(pt_file, map_location=self.device, weights_only=True)
            i = list(read_codes(idx_file))
            print(f"INFO\tloaded {len(e)} reference embeddings")
            if terminals_only:
                ei = [
                    (u, v)
                    for u, v in zip(e, i)
                    if icd.is_valid_item(v) and icd.is_leaf(v)
                ]
                print(f"INFO\treduced to {len(ei)} terminals")
            else:
                ei = [(u, v) for u, v in zip(e, i) if icd.is_valid_item(v)]

            icd10_e.extend([torch.tensor(u) for u, v in ei])
            icd10_i.extend([v for u, v in ei])
            # load just one shard for debugging purpose
            if self.cfg.get("debug", False):
                break
        self.icd10_e = icd10_e
        self.icd10_i = icd10_i

    def _load_icd10(self, num_shards: int = 10, terminals_only: bool = True):
        # Load reference embeddings and codes
        icd10_dir = self.paths["icd10_e"]
        if self.reducer is not None:
            return self._load_icd10_one_shard(icd10_dir, terminals_only=terminals_only)
        return self._load_icd10_multiple_shards(icd10_dir, num_shards, terminals_only)


# Usage in benchmark:
# self.data = BenchmarkData(cfg)
# Access:
#   embeddings and metadata of test set: data.test_e, data.test_i
#   embeddings and metadata of train set: data.train_e, data.train_i
#   embeddings and metadata of icd10: data.icd10_e and data.icd10_i
