"""
vae.py
------
Extracts VAE features from 3D MRI scans using CogVideoX-5b and saves them to /tmp.

Usage:
    python vae.py --train_csv /path/to/train.csv --train_data_dir /path/to/train/ --test_csv /path/to/test.csv --test_data_dir /path/to/test/ --train_num 4118 --test_num 582

Dependencies:
    pip install torch diffusers numpy pandas nibabel
"""

import torch
from diffusers import AutoencoderKLCogVideoX
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import os


def load_mri(idx: int, data_dir: str, csv_path: str = '/datasets/adni_3d/adni_metadata.csv') -> torch.Tensor:
    """
    Load and preprocess a single MRI scan.
    Args:
        idx: Index of the scan in the metadata CSV.
        csv_path: Path to the metadata CSV file.
        data_dir: Directory containing MRI files.
    Returns:
        Preprocessed MRI as a torch.Tensor.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, usecols=['fname'], header=0)
    fnames = df['fname'].tolist()
    fname = fnames[idx]
    img_path = os.path.join(data_dir, fname)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"MRI file not found: {img_path}")
    img = nib.load(img_path)
    data = img.get_fdata()
    max_value = np.percentile(data, 95)
    min_value = np.percentile(data, 5)
    data = np.where(data <= max_value, data, max_value)
    data = np.where(data <= min_value, 0., data)
    data = (data / max_value) * 2 - 1  # -1,1
    img = np.ones((144, 176, 144)) * data.min()
    img[3:3+138, :, 3:3+138] = data
    img = np.transpose(img, (2, 1, 0))
    data = torch.from_numpy(img[None, None, :, :, :]).float()
    data = data.repeat(1, 3, 1, 1, 1)
    assert data.shape == (1, 3, 144, 176, 144)
    return data


def extract_features(
    vae,
    num_samples: int,
    csv_path: str,
    data_dir: str,
    output_path: str
):
    """
    Extract VAE features for a dataset and save to output_path.
    """
    feature_shape = (num_samples, 1, 16, 36, 22, 18)
    features = torch.empty(feature_shape).to('cuda')
    for i in range(num_samples):
        try:
            mri = load_mri(i, csv_path=csv_path, data_dir=data_dir).to('cuda')
            posterior = vae.encode(mri.to(torch.bfloat16)).latent_dist
            z = posterior.mode()
            assert z.shape == (1, 16, 36, 22, 18)
            features[i] = z
            print(f'\rPropagating idx {i}', end='', flush=True)
        except Exception as e:
            print(f"\nError processing index {i}: {e}")
    features = features.cpu().numpy()
    np.save(output_path, features)
    print(f'\nSaved features to {output_path}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract VAE features from 3D MRI scans using CogVideoX-5b.")
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training metadata CSV')
    parser.add_argument('--train_data_dir', type=str, required=True, help='Directory with training MRI files')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test metadata CSV')
    parser.add_argument('--test_data_dir', type=str, required=True, help='Directory with test MRI files')
    parser.add_argument('--train_num', type=int, required=True, help='Number of training samples')
    parser.add_argument('--test_num', type=int, required=True, help='Number of test samples')
    parser.add_argument('--train_output', type=str, default='/tmp/train_feature_extracted_by_CogVideoX-5b.npy', help='Output path for train features')
    parser.add_argument('--test_output', type=str, default='/tmp/test_feature_extracted_by_CogVideoX-5b.npy', help='Output path for test features')
    args = parser.parse_args()

    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b", subfolder="vae", torch_dtype=torch.bfloat16
    ).to('cuda')

    with torch.no_grad():
        extract_features(
            vae,
            num_samples=args.train_num,
            csv_path=args.train_csv,
            data_dir=args.train_data_dir,
            output_path=args.train_output
        )
        extract_features(
            vae,
            num_samples=args.test_num,
            csv_path=args.test_csv,
            data_dir=args.test_data_dir,
            output_path=args.test_output
        )