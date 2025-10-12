import subprocess
import torch


def get_repo_root():
    return (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )


def load_tensor(pt_file, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "has_mps") and torch.has_mps:
            device = "mps"
        else:
            device = "cpu"
    try:
        return torch.load(
            pt_file, map_location=torch.device(device), weights_only=False
        )
    except RuntimeError:
        try:
            return torch.load(
                pt_file, map_location=torch.device(device), _legacy_load=True
            )
        except:
            with open(pt_file, "rb") as f:
                return torch.load(f, map_location=torch.device(device))
