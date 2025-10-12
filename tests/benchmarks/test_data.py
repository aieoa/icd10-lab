# icd10-lab/tests/unittests/benchmarks/test_data.py

"""
Tests primarily presence of te3l embeddings for codiesp with
sentence-level chunking stored under ./data/ in repository root.
"""
import os
from pandas import DataFrame
from pathlib import Path
import pytest
import sys
from unittest.mock import patch

sys.path.append("..")

from src.benchmarks.data import BenchmarkData
from src.utils.sysops import get_repo_root
from tests.common import check_config_keys


def create_config(data_dir, benchmark_str, chunking_strategy):
    cfg = {
        "name": "test",
        "benchmark": benchmark_str,
        "data_dir": data_dir,
        "solver": "knn",
        "solver_params": {"similarity": "cosine"},
        "embedding_model": "te3l",
        "icd10_params": {"aggregate_method": "leaf", "terminals_only": True},
        "chunking_strategy": chunking_strategy,
        "reducer": None,
        "reducer_params": {"n_components": None},
        "subset": {"test": [], "train": []},
    }
    check_config_keys(cfg)
    return cfg


@pytest.fixture
def tmp_config(tmp_path):
    benchmark_str = "codiesp_en"
    chunking_strategy = "sent"
    data_dir = tmp_path / "data"
    (data_dir / "03_derived/codiesp_en/test/te3l/sent/original").mkdir(parents=True)
    (data_dir / "03_derived/codiesp_en/train/te3l/sent/original").mkdir(parents=True)
    (data_dir / "03_derived/icd10/te3l/leaf/original").mkdir(parents=True)
    (data_dir / "02_processed/codiesp_en/test/sent").mkdir(parents=True)
    (data_dir / "02_processed/codiesp_en/train/sent").mkdir(parents=True)
    return create_config(data_dir, benchmark_str, chunking_strategy)


@pytest.fixture
def default_config(tmp_path):
    benchmark_str = "codiesp_en"
    chunking_strategy = "sent"
    data_dir = Path(get_repo_root()) / "data"
    return create_config(data_dir, benchmark_str, chunking_strategy)


def test_setup_paths(tmp_config):
    cfg = tmp_config.copy()
    with patch.object(BenchmarkData, "_load_indices", return_value=None), patch.object(
        BenchmarkData, "_load_embeddings", return_value=None
    ), patch.object(BenchmarkData, "_load_icd10", return_value=None):
        data = BenchmarkData(cfg)
        for subset in ["test", "train", "icd10"]:
            for data_type in ["e", "i"]:
                path_key = f"{subset}_{data_type}"
                assert path_key in data.paths
                assert os.path.exists(data.paths[path_key])


def test_call_load_indices(tmp_config):
    cfg = tmp_config.copy()
    with patch.object(
        BenchmarkData, "_load_indices", return_value=None
    ) as mock_load_indices, patch.object(
        BenchmarkData, "_load_embeddings", return_value=None
    ), patch.object(
        BenchmarkData, "_load_icd10", return_value=None
    ):
        data = BenchmarkData(cfg)
        assert mock_load_indices.called


def test_call_load_embeddings(tmp_config):
    cfg = tmp_config.copy()
    with patch.object(BenchmarkData, "_load_indices", return_value=None), patch.object(
        BenchmarkData, "_load_embeddings", return_value=None
    ) as mock_load_embed, patch.object(BenchmarkData, "_load_icd10", return_value=None):
        data = BenchmarkData(cfg)
        assert mock_load_embed.called


def test_call_load_icd10(tmp_config):
    cfg = tmp_config.copy()
    with patch.object(BenchmarkData, "_load_indices", return_value=None), patch.object(
        BenchmarkData, "_load_embeddings", return_value=None
    ), patch.object(BenchmarkData, "_load_icd10", return_value=None) as mock_load_icd10:
        data = BenchmarkData(cfg)
        assert mock_load_icd10.called


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Skipping index loading test in CI environment."
)
def test_load_indices(default_config):
    # assumes data files are present in the repo data directory for
    # benchmark "codiesp_en", te3l embedding, and chunking strategy "sent"
    cfg = default_config.copy()
    data_dir = Path(get_repo_root()) / "data"
    test_i = data_dir / "02_processed/codiesp_en/test/sent"
    train_i = data_dir / "02_processed/codiesp_en/train/sent"
    icd10_i = data_dir / "03_derived/icd10/te3l/leaf/original"

    assert os.path.exists(test_i)
    assert os.path.exists(train_i)
    assert os.path.exists(icd10_i)
    assert len(os.listdir(test_i))
    assert len(os.listdir(train_i))
    assert len(os.listdir(icd10_i))

    with patch.object(
        BenchmarkData, "_load_embeddings", return_value=None
    ), patch.object(BenchmarkData, "_load_icd10", return_value=None):
        data = BenchmarkData(cfg)
        assert isinstance(data.test_i, DataFrame)
        assert isinstance(data.train_i, DataFrame)
        assert not data.test_i.empty
        assert not data.train_i.empty


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Skipping data loading test in CI environment."
)
def test_load_embeddings(default_config):
    cfg = default_config.copy()
    data_dir = Path(get_repo_root()) / "data"
    test_e = data_dir / "03_derived/codiesp_en/test/te3l/sent/original"
    train_e = data_dir / "03_derived/codiesp_en/train/te3l/sent/original"
    icd10_e = data_dir / "03_derived/icd10/te3l/leaf/original"

    with patch.object(BenchmarkData, "_load_indices", return_value=None):
        data = BenchmarkData(cfg)
        assert os.path.exists(test_e)
        assert os.path.exists(train_e)
        assert os.path.exists(icd10_e)
        assert len(os.listdir(test_e))
        assert len(os.listdir(train_e))
        assert len(os.listdir(icd10_e))


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Skipping data loading test in CI environment."
)
def test_data_consistency(default_config):
    # check if dimensions of index and embeddings as expected
    cfg = default_config.copy()
    data = BenchmarkData(cfg)
    # index and embedings should have same number of entries
    data.test_i.shape[0] == len(data.test_e)
    data.train_i.shape[0] == len(data.train_e)
    # vector dims in test and train should be identical
    data.test_e.shape[1] == data.train_e.shape[1]


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Skipping data loading test in CI environment."
)
def test_reducer_umap_consistency(default_config):
    # check if reducer is set to umap when specified in config
    n_components = 10
    cfg = default_config.copy()
    # measure original dimensions
    with patch.object(BenchmarkData, "_load_icd10", return_value=None):
        data = BenchmarkData(cfg)
        assert data.reducer is None
        test_num_entries = data.test_e.shape[0]
        train_num_entries = data.train_e.shape[0]
        test_dim = data.test_e.shape[1]
        train_dim = data.train_e.shape[1]

    # now set reducer to umap with 10 components
    cfg["reducer"] = "umap"
    cfg["reducer_params"]["n_components"] = n_components

    with patch.object(BenchmarkData, "_load_icd10", return_value=None):
        data = BenchmarkData(cfg)
        assert data.reducer is not None
        # index and embedings should have same number of entries
        data.test_i.shape[0] == len(data.test_e)
        data.train_i.shape[0] == len(data.train_e)
        # number of entries should be unchanged
        data.test_e.shape[0] == test_num_entries
        data.train_e.shape[0] == train_num_entries
        # but dimension should be reduced to 10
        data.test_e.shape[1] == n_components
        data.train_e.shape[1] == n_components
        data.test_e.shape[1] < test_dim
        data.train_e.shape[1] < train_dim


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Skipping data loading test in CI environment."
)
def test_reducer_umap_consistency_icd10(default_config):
    # check if reducer is set to umap when specified in config
    n_components = 10
    cfg = default_config.copy()
    # measure original dimensions
    with patch.object(BenchmarkData, "_load_indices", return_value=None), patch.object(
        BenchmarkData, "_load_indices", return_value=None
    ):
        data = BenchmarkData(cfg)
        assert data.reducer is None
        num_entries = len(data.icd10_e)
        dim = data.icd10_e[0].size(dim=0)

    # now set reducer to umap with 10 components
    cfg["reducer"] = "umap"
    cfg["reducer_params"]["n_components"] = n_components

    with patch.object(BenchmarkData, "_load_indices", return_value=None), patch.object(
        BenchmarkData, "_load_indices", return_value=None
    ):
        data = BenchmarkData(cfg)
        assert data.reducer is not None
        # index and embedings should have same number of entries
        assert len(data.icd10_i) == len(data.icd10_e)
        # number of entries should be unchanged
        assert len(data.icd10_e) == num_entries
        # but dimension should be reduced to n_components
        data.icd10_e[0].size(dim=0) == n_components
        data.icd10_e[0].size(dim=0) < dim
