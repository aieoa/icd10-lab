# src/benchmarks/data.py

from datetime import timedelta
import logging
import numpy as np
import os
import torch
import pandas as pd
from pathlib import Path
import simple_icd_10_cm as icd
import time
from typing import List, Dict, Any

# import src.utils.sysops as sysops

logger = logging.getLogger(__name__)


class BenchmarkData:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = cfg.get("device", None)
        self.benchmark_str = f"{cfg['benchmark'].lower()}_{cfg['language'].lower()}"
        self.data_dir = cfg["data_dir"]  # base data directory
        self.chunking = cfg["chunking"]
        self.text_encoder = cfg["text_encoder"]
        self.reducer = cfg.get("reducer", None)
        self.reducer_params = cfg.get("reducer_params", {})
        self.icd10_params = cfg["icd10_params"]
        self.terminals_only = self.icd10_params.get("terminals_only", True)
        self.subset = {
            "test": cfg.get("subset", {}).get("test", []),
            "train": cfg.get("subset", {}).get("train", []),
        }

        self._setup_paths()
        self._set_device()

        # Load indices/labels for train/test/reference
        self.train_i = self._load_indices("train", subset=self.subset["train"])
        self.test_i = self._load_indices("test", subset=self.subset["test"])

        # Load embeddings for train/test/reference
        self.train_e = self._load_embeddings("train", subset=self.subset["train"])
        self.test_e = self._load_embeddings("test", subset=self.subset["test"])

        # Load reference embeddings and codes
        self._load_icd10()

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
        benchmark_subdir = Path(self.text_encoder) / self.chunking / subdir
        

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
            / self.text_encoder
            / self.icd10_params["aggregate_method"]
            / subdir,
            "test_i": Path(self.data_dir)
            / "02_processed"
            / self.benchmark_str
            / "test"
            / self.chunking,
            "train_i": Path(self.data_dir)
            / "02_processed"
            / self.benchmark_str
            / "train"
            / self.chunking,
            "icd10_i": Path(self.data_dir)
            / "03_derived"
            / "icd10"
            / self.text_encoder
            / self.icd10_params["aggregate_method"]
            / subdir,
        }
        for path in self.paths.values():
            assert os.path.exists(
                path
            ), f"Path does not exist: {path}, correct or precompute embeddings."


    def _load_indices(self, split: str, subset: List = None):
        # todo: load here dvc versioned data set with associated index
        dataset_dir = self.paths[f"{split}_i"]
        if self.chunking == "full":  # txt files
            index_files = [f for f in os.listdir(dataset_dir) if f.endswith(".txt")]
        else:  # csv files
            index_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
        indices = pd.DataFrame()
        for fname in index_files:
            report_id = Path(fname).stem.split("_")[0]

            if report_id == "S1139-76322014000500011-1":
                continue
            if len(subset) and report_id not in subset:
                continue
            i = pd.read_csv(os.path.join(dataset_dir, fname))
            if "report_id" not in i.columns:
                i["report_id"] = report_id
            indices = pd.concat([indices, i], axis=0, ignore_index=True)
        return None if indices.empty else indices

    def _load_embeddings(self, split: str, subset: List = None):
        embedding_dir = self.paths[f"{split}_e"]
        embedding_list = []
        assert len(os.listdir(embedding_dir)), f"No embeddings found in {embedding_dir}"
        for npy_path in embedding_dir.glob("*.npy"):
            report_id = npy_path.stem.split("_")[0]
            if report_id == "S1139-76322014000500011-1":
                continue
            if subset and report_id not in subset:
                continue
            e = np.load(npy_path)
            embedding_list.append(e)
        if not embedding_list:
            raise ValueError(f"No valid embeddings found in {embedding_dir} for given subset.")
        embedding_array = np.concatenate(embedding_list)
        return embedding_array

    
    def _load_icd10(self):
        """
        Loads all .npy embedding shards in the given directory
        and their corresponding .pt.idx index files,
        filters according to ICD-10 criteria, concatenates,
        and sets self.icd10_e and self.icd10_i.
        """
        icd10_dir = self.paths["icd10_e"]
    
        embedding_list = []
        index_list = []
        
        for npy_path in sorted(icd10_dir.glob("*.npy")):
            idx_path = npy_path.with_suffix(".pt.idx")
            if not idx_path.exists():
                print(f"Index file missing for {npy_path.name}, skipping shard.")
                continue
            
            # Load embedding and index
            icd10_e = np.load(npy_path)
            with open(idx_path, "r") as f:
                icd10_i = pd.Series([line.strip() for line in f])

            # Apply mask filtering
            mask = icd10_i.apply(icd.is_valid_item)
            if self.terminals_only:
                mask &= icd10_i[mask].apply(icd.is_leaf)

            # Keep only filtered embeddings and indices
            filtered_e = icd10_e[mask.values]
            filtered_i = icd10_i[mask].to_numpy()

            # Append to total list
            embedding_list.append(filtered_e)
            index_list.append(filtered_i)

        if not embedding_list:
            raise ValueError(f"No valid embeddings found in {icd10_dir}.")
        
        # Concatenate all valid embeddings and indices
        self.icd10_e = np.concatenate(embedding_list, axis=0)
        self.icd10_i = np.concatenate(index_list, axis=0)
