from pathlib import Path
import yaml

from src.utils.sysops import get_repo_root


def check_config_keys(generated_cfg):
    example_cfg_path = Path(get_repo_root()) / "config.example.yaml"
    # Check that keys config.example.yaml are superset of cfg
    # generated for testing purposes.
    with open(example_cfg_path, "r") as f:
        example_cfg = yaml.safe_load(f)
    unknown_keys = set(generated_cfg.keys()) - set(example_cfg.keys())
    assert not unknown_keys, f"Missing keys in generated config: {unknown_keys}"
