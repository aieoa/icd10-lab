import joblib
import numpy as np
import os
from pathlib import Path
from scipy.sparse import load_npz
import umap
import argparse


def go(X_train, X_test, model_path, n_components:int=10, seed:int=81):
    umap_transformer = umap.UMAP(
        n_components=n_components,
        random_state=seed
    )
    
    # Transform training data
    print(f"Fitting and transforming training data...")
    X_train_umap = umap_transformer.fit_transform(X_train)
    
    # Save the fitted model
    joblib.dump(umap_transformer, model_path)
    print(f"Saved UMAP model to {model_path}")
    
    # Transform test data
    print(f"Transforming test data...")
    X_test_umap = umap_transformer.transform(X_test)
    # np.save(test_output.with_suffix('.npy'), X_test_umap)
    return X_train_umap, X_test_umap


def parse_args():
    parser = argparse.ArgumentParser(description='Apply UMAP transformation to train and test data')
    parser.add_argument('--train', type=str, required=True,
                      help='Path to training data (.npy file)')
    parser.add_argument('--test', type=str, required=True,
                      help='Path to test data (.npy file)')
    parser.add_argument('-n', '--n-components', type=int, default=50,
                      help='Number of components for UMAP (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    return parser.parse_args()

def main():
    # os.environ["OMP_NUM_THREADS"] = "1"

    args = parse_args()
    
    # Convert file paths to Path objects
    train_file = Path(args.train)
    test_file = Path(args.test)
    
    outdir = train_file.parent
    # Create output file paths
    train_output = outdir / f"{train_file.stem}_umap.npz"
    test_output = outdir / f"{test_file.stem}_umap.npz"
    
    # Load input data
    X_train = load_npz(train_file)
    X_test = load_npz(test_file)
    
    # Initialize and fit UMAP
    umap_transformer = umap.UMAP(
        n_components=args.n_components,
        random_state=args.seed
    )
    
    # Transform training data
    print(f"Fitting and transforming training data...")
    X_train_umap = umap_transformer.fit_transform(X_train)
    # Save dense output
    np.save(train_output.with_suffix('.npy'), X_train_umap)

    print(f"Saved transformed training data to {train_output}")
    
    # Save the fitted model
    umap_dir = outdir / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)
    model_path = umap_dir / f"umap_model_{args.n_components}.joblib"
    joblib.dump(umap_transformer, model_path)
    print(f"Saved UMAP model to {model_path}")
    
    # Transform test data
    print(f"Transforming test data...")
    X_test_umap = umap_transformer.transform(X_test)
    np.save(test_output.with_suffix('.npy'), X_test_umap)

    print(f"Saved transformed test data to {test_output}")

if __name__ == "__main__":
    main()