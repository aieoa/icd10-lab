# src/utils/converter.py

import torch
import numpy as np
from pathlib import Path


def convert_to_numpy(input_dir, output_dir=None):
    """Convert files into numpy format.
    Expect a directory of files storing tensors in pytorch format.
    Each file will be converted to npy and stored under same name.
    """
    input_dir = Path(input_dir)
    output_dir = input_dir if output_dir is None else output_dir

    print(f"INFO\tConverting pytorch tensors in\t{input_dir}")
    for pt_path in input_dir.glob("*.pt"):
        try:
            embeddings = torch.load(pt_path, map_location="cpu")
        except:
            print(f"WARNING\tskipping (torch.load fails) {pt_path}")
            continue
        # Convert to numpy array: handle Tensor, list of lists, etc.
        if isinstance(embeddings, torch.Tensor):
            arr = embeddings.cpu().numpy()
        else:  # e.g. list of lists, list of tensors, etc.
            arr = np.array(embeddings)
        npy_path = pt_path.with_suffix(".npy")
        np.save(npy_path, arr)
    print(f"INFO\tDone")
