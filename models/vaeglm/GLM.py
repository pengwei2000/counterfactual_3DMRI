"""GLM.py
-------
General Linear Model utilities for feature residualization in 3D MRI counterfactual benchmarks.

Adapted from https://arxiv.org/pdf/2409.05585

Functions:
    - norm_list: Normalize a list to z-scores.
    - convert_metadata: Normalize columns of metadata.
    - get_kernel: Compute kernel and metadata matrix for GLM.
    - GLM: Compute residuals and betas for features given metadata.
    - compute_GLM: Example function to compute and save GLM results.
    - compute_residual: Example function to compute and save residuals for test data.

Usage:
    python GLM.py --train_metadata /tmp/train_metadata --train_feature /tmp/train_feature_extracted_by_CogVideoX-5b.npy --test_metadata /tmp/test_metadata --test_feature /tmp/test_feature_extracted_by_CogVideoX-5b.npy --beta /tmp/Beta --train_residual /tmp/train_Residual --test_residual /tmp/test_Residual

Dependencies:
    pip install torch numpy
"""

import torch
import numpy as np
import os

def norm_list(data: np.ndarray) -> list:
    """
    Normalize a 1D numpy array to z-scores.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    if std_dev == 0:
        raise ValueError("Standard deviation is zero; cannot normalize.")
    return ((data - mean) / std_dev).tolist()

def convert_metadata(Metadata: np.ndarray) -> list:
    """
    Normalize each column of the metadata array.
    """
    Metadata_New = []
    for i in range(Metadata.shape[-1]):
        vol = norm_list(Metadata[:, i])
        Metadata_New.append(vol)
    return Metadata_New

def get_kernel(Metadata: list) -> tuple:
    """
    Build metadata matrix and compute kernel for GLM.
    """
    N = len(Metadata[0])
    metadata = np.ones((N, 8))
    metadata[:, 1] = Metadata[0]  # sex
    metadata[:, 2] = Metadata[1]  # age
    metadata[:, 3] = Metadata[2]  # diagnosis
    metadata[:, 4] = Metadata[3]  # svol
    metadata[:, 5] = Metadata[4]  # Frontal_raw
    metadata[:, 6] = Metadata[5]  # Insula_raw
    metadata[:, 7] = Metadata[6]  # Parietal_raw
    cf_kernel = torch.tensor(np.linalg.inv(np.transpose(metadata).dot(metadata))).float()
    return cf_kernel, torch.from_numpy(metadata).float()

def GLM(X_feature: torch.Tensor, Metadata: list) -> tuple:
    """
    Compute GLM residuals and betas for features given metadata.
    Args:
        X_feature: (N, D) feature tensor
        Metadata: list of normalized metadata columns
    Returns:
        residual, Beta, Metadata_matrix
    """
    cf_kernel, Metadata_matrix = get_kernel(Metadata)
    Meta_T = torch.transpose(Metadata_matrix, 0, 1)
    pinv = torch.mm(cf_kernel, Meta_T)
    Beta = torch.mm(pinv, X_feature)
    X_r = torch.mm(Metadata_matrix, Beta)
    residual = X_feature - X_r
    residual = residual.reshape(X_feature.shape)
    return residual, Beta, Metadata_matrix

def compute_GLM(
    metadata_path: str,
    feature_path: str,
    beta_save_path: str,
    residual_save_path: str
):
    """
    Compute and save GLM betas and residuals for training data.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    Metadata = torch.load(metadata_path).numpy()
    X_feature = np.load(feature_path)
    X_feature = X_feature.reshape(X_feature.shape[0], -1)
    X_feature = torch.from_numpy(X_feature)
    Metadata = convert_metadata(Metadata)
    residual, Beta, Metadata_matrix = GLM(X_feature, Metadata)
    torch.save(Beta, beta_save_path)
    torch.save(residual, residual_save_path)

def compute_residual(
    metadata_path: str,
    feature_path: str,
    beta_path: str,
    residual_save_path: str
):
    """
    Compute and save residuals for test data using precomputed Beta.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    if not os.path.exists(beta_path):
        raise FileNotFoundError(f"Beta file not found: {beta_path}")
    Metadata = torch.load(metadata_path).numpy()
    X_feature = np.load(feature_path)
    Beta_from_train = torch.load(beta_path)
    X_feature = X_feature.reshape(X_feature.shape[0], -1)
    X_feature = torch.from_numpy(X_feature)
    Metadata = convert_metadata(Metadata)
    _, Metadata_matrix = get_kernel(Metadata)
    residual = X_feature - torch.mm(Metadata_matrix, Beta_from_train)
    torch.save(residual, residual_save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute and save GLM betas and residuals for 3D MRI features.")
    parser.add_argument('--train_metadata', type=str, required=True, help='Path to training metadata (torch file)')
    parser.add_argument('--train_feature', type=str, required=True, help='Path to training features (npy file)')
    parser.add_argument('--test_metadata', type=str, required=True, help='Path to test metadata (torch file)')
    parser.add_argument('--test_feature', type=str, required=True, help='Path to test features (npy file)')
    parser.add_argument('--beta', type=str, default='/tmp/Beta', help='Path to save Beta (default: /tmp/Beta)')
    parser.add_argument('--train_residual', type=str, default='/tmp/train_residual', help='Path to save train residual (default: /tmp/train_residual)')
    parser.add_argument('--test_residual', type=str, default='/tmp/test_residual', help='Path to save test residual (default: /tmp/test_residual)')
    args = parser.parse_args()

    try:
        compute_GLM(
            metadata_path=args.train_metadata,
            feature_path=args.train_feature,
            beta_save_path=args.beta,
            residual_save_path=args.train_residual
        )
        compute_residual(
            metadata_path=args.test_metadata,
            feature_path=args.test_feature,
            beta_path=args.beta,
            residual_save_path=args.test_residual
        )
        print("GLM computation and saving completed successfully.")
    except Exception as e:
        print(f"Error: {e}")