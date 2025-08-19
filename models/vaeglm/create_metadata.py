"""
create_metadata.py
------------------
Creates and saves normalized metadata tensors for VAE-GLM experiments. All output is saved to /tmp for reproducibility.

Usage:
    python create_metadata.py --train_idx datasets\adni_3d\train_idx.npy --test_idx datasets\adni_3d\test_idx.npy --csv_path datasets\adni_3d\adni_metadata.csv

Dependencies:
    pip install torch numpy pandas nibabel
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
import os
import argparse

MIN_MAX = {
    'image': [-1.0, 1.0],
    'age': [55.25, 96.],
    'fro': [136061.0, 212592.0],
    'par': [88797.0, 139170.0],
    'tem': [94958.0, 175299.0],
    'occ': [36153.0, 70856.0],
    'cin': [20419.0, 34470.0],
    'ins': [12401.0, 22950.0],
    'ven': [9527.0, 157058.0]
}

def normalize(value, name):
    if name not in MIN_MAX or name == 'image':
        return value
    # [min,max] -> [0,1]
    value = (value - MIN_MAX[name][0]) / (MIN_MAX[name][1] - MIN_MAX[name][0])
    value = value * 2 - 1  # [0,1] -> [-1,1]
    return value

def unnormalize(value, name):
    if name not in MIN_MAX or name == 'image':
        return value
    value = (value + 1) / 2  # [-1,1] -> [0,1]
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) + MIN_MAX[name][0]
    return value

class ADNI3d(Dataset):
    def __init__(self, attribute_size, split='train', normalize_=True, transform=None, transform_cls=None, train_idx=None, test_idx=None, csv_path=None, data_dir=None):
        super().__init__()
        self.split = split
        self.transform = transform
        self.transform_cls = transform_cls
        self.has_valid_set = True
        self.train_idx = np.load(train_idx).tolist()
        self.test_idx = np.load(test_idx).tolist()
        assert len(self.test_idx) + len(self.train_idx) == 4578, f'The train and test set should be 4578'
        self.csv_path = csv_path
        self.data_dir = data_dir

        self.attributes = self.load_attributes(attributes=attribute_size.keys())

        if normalize_:
            self.attributes = {attr: normalize(torch.tensor(values, dtype=torch.float32), attr) for attr, values in self.attributes.items()}
        else:
            self.attributes = {attr: torch.tensor(values, dtype=torch.float32) for attr, values in self.attributes.items()}
        if "sex" in attribute_size:
            self.attributes['sex'] = F.one_hot(self.attributes['sex'].to(torch.long), num_classes=2)
        if "diagnosis" in attribute_size:
            self.attributes['diagnosis'] = F.one_hot(self.attributes['diagnosis'].to(torch.long), num_classes=5)

        self.attrs = torch.cat([
            self.attributes[attr].unsqueeze(1) if len(self.attributes[attr].shape) == 1 else self.attributes[attr]
            for attr in attribute_size.keys()
        ], dim=1)
        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.attributes.items()}

        bins = np.array([-1, -0.1, 1])
        self.bins = {}
        for attr, values in self.attributes.items():
            if attr not in ["sex", "diagnosis"]:
                data = values.numpy()
                assert data.max() <= 1, 'Please check the file dataset.py, ADNI3d.bins wants to put attr values into bins from -1 to 1, but the attr value is not normalized to -1 to 1.'
                digitized = np.digitize(data, bins)
                self.bins[attr] = [data[digitized == i].mean() for i in range(1, len(bins))]
                for i in range(1, len(bins)):
                    if len(data[digitized == i]) == 0:
                        print(attr, i)
    def load_attributes(self, attributes):
        usecols = [att for att in attributes]
        df = pd.read_csv(self.csv_path, usecols=usecols, header=0)
        if self.split == 'train':
            df = df.loc[self.train_idx]
        elif self.split == 'valid' or self.split == "test":
            df = df.loc[self.test_idx]
        attribute_dict = {}
        for att in attributes:
            if att == 'diagnosis':
                df[att] = df[att].replace('AD', 4)
                df[att] = df[att].replace('LMCI', 3)
                df[att] = df[att].replace('EMCI', 2)
                df[att] = df[att].replace('SMC', 1)
                df[att] = df[att].replace('CN', 0)
            elif att == "sex":
                df[att] = df[att].replace('M', 1)
                df[att] = df[att].replace('F', 0)
            attribute_dict[att] = df[att].tolist()
        return attribute_dict

    def load_images(self, idx):
        csv_path = 'datasets/adni_3d/adni_metadata.csv'
        df = pd.read_csv(csv_path, usecols=['fname'], header=0)
        if self.split == 'train':
            df = df.loc[self.train_idx]
        elif self.split == 'valid' or self.split == "test":
            df = df.loc[self.test_idx]
        fnames = df['fname'].tolist()

        images = []
        fname = fnames[idx]
        # print(fname)
        image_path = Path(self.data_dir + fname)
        img = nib.load(image_path)
        data = img.get_fdata(dtype=np.float32)  # (138,176,138)
        max_value = np.percentile(data, 95)
        min_value = np.percentile(data, 5)
        data = np.where(data <= max_value, data, max_value)
        data = np.where(data <= min_value, 0., data)
        data = data / max_value  # normalize to [0,1]
        data = data * 2 - 1  # normalize to [-1,1]
        pad_zero = -np.ones((144, 176, 144), dtype=np.float32)
        pad_zero[3:3 + 138, :, 3:3 + 138] = data
        assert pad_zero.max() <= 1, "After resize, the value is out of (-1,1)"
        assert pad_zero.min() >= -1, "After resize, the value is out of (-1,1)"
        assert pad_zero.shape[0] == 144
        assert pad_zero.shape[1] == 176
        images.append(pad_zero)
        return np.array(images)

    def __len__(self):
        return self.attrs.shape[0]

    def __getitem__(self, idx):
        # self.images = 0
        print(self.images.shape)  # (1, 144, 176, 144)

        if self.transform:
            return self.transform(self.images, self.attrs[idx])

        if self.transform_cls:
            return self.transform_cls(self.images), self.attrs[idx]

        # print(self.attrs[idx])


def save_metadata(attribute_size, split, batch_size, save_path, train_idx, test_idx, csv_path, data_dir):
    data = ADNI3d(attribute_size=attribute_size, split=split, train_idx=train_idx, test_idx=test_idx, csv_path=csv_path, data_dir=data_dir)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in data_loader:
        print("Data shape:", batch[0].shape)
        print("Labels shape:", batch[1].shape)
        torch.save(batch[1], save_path)
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and save normalized metadata tensors for VAE-GLM experiments.")
    parser.add_argument('--train_idx', type=str, required=True, help='Path to train_idx.npy')
    parser.add_argument('--test_idx', type=str, required=True, help='Path to test_idx.npy')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to adni_metadata.csv')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing MRI files')
    parser.add_argument('--train_batch', type=int, default=4118, help='Train batch size (default: 4118)')
    parser.add_argument('--test_batch', type=int, default=460, help='Test batch size (default: 460)')
    parser.add_argument('--train_output', type=str, default='/tmp/train_metadata', help='Output path for train metadata')
    parser.add_argument('--test_output', type=str, default='/tmp/test_metadata', help='Output path for test metadata')
    args = parser.parse_args()

    attribute_size = {"fro": 1, "par": 1, "tem": 1, "occ": 1, "cin": 1, "ins": 1, "ven": 1}
    save_metadata(attribute_size, "train", args.train_batch, args.train_output,
                 train_idx=args.train_idx, test_idx=args.test_idx, csv_path=args.csv_path, data_dir=args.data_dir)
    save_metadata(attribute_size, "test", args.test_batch, args.test_output,
                 train_idx=args.train_idx, test_idx=args.test_idx, csv_path=args.csv_path, data_dir=args.data_dir)
