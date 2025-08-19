"""
NCANDA Dataset Loader
---------------------
PyTorch Dataset and utilities for the NCANDA dataset, with normalization and image loading.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
from typing import Dict, Any

MIN_MAX = {
    'image': [-1.0, 1.0],
    'age': [12.0, 28.3],
    'fro': [176848., 239506.1],
    'par':[111514.,150783.9],
    'tem':[116488.9, 146967.2],
    'occ':[42157.1, 67115.9],
    'cin':[24494.5, 32582.8],
    'ins':[13953.9, 18298.1],
    'ven':[5735.7, 48218.4]
}

def normalize(value: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a value to [-1, 1] based on MIN_MAX for the given attribute name."""
    if name not in MIN_MAX or name == 'image':
        return value
    # [min,max] -> [0,1]
    value = (value - MIN_MAX[name][0]) / (MIN_MAX[name][1] - MIN_MAX[name][0])
    value = value * 2 - 1  # [0,1] -> [-1,1]
    return value

def unnormalize(value: torch.Tensor, name: str) -> torch.Tensor:
    """Unnormalize a value from [-1, 1] to original scale based on MIN_MAX for the given attribute name."""
    if name not in MIN_MAX or name == 'image':
        return value
    value = (value + 1) / 2  # [-1,1] -> [0,1]
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value

class NCANDA(Dataset):
    """
    PyTorch Dataset for the NCANDA dataset.
    Args:
        attribute_size: Dict of attribute names and their sizes.
        split: 'train' or 'test'.
        normalize_: Whether to normalize attributes.
        transform: Optional transform for images.
        transform_cls: Optional transform for class labels.
    """
    def __init__(self, attribute_size: Dict[str, int], split: str = 'test', normalize_: bool = True, transform: Any = None, transform_cls: Any = None):
        super().__init__()
        self.split = split
        self.transform = transform
        self.transform_cls = transform_cls
        self.has_valid_set = True

        self.attributes = self.load_attributes(attributes=attribute_size.keys())

        if normalize_:
            self.attributes = {attr: normalize(torch.tensor(values, dtype=torch.float32), attr) for attr, values in self.attributes.items()}
        else:
            self.attributes = {attr: torch.tensor(values, dtype=torch.float32) for attr, values in self.attributes.items()}
        
        # change to one-hot representation
        if 'sex' in self.attributes:
            self.attributes['sex'] = F.one_hot(self.attributes['sex'].to(torch.long), num_classes=2)
        if 'diagnosis' in self.attributes:
            self.attributes['diagnosis'] = F.one_hot(self.attributes['diagnosis'].to(torch.long), num_classes=4)
        
        self.attrs = torch.cat([self.attributes[attr].unsqueeze(1) if len(self.attributes[attr].shape) == 1 else self.attributes[attr]
                                for attr in attribute_size.keys()], dim=1)
        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.attributes.items()}

        bins = np.array([-1, 0 ,1])
        biased_bins = np.array([-1, -0.35, 1])
        self.bins = {}
        for attr, values in self.attributes.items():
            if attr not in ["sex", "diagnosis"]:
                data = values.numpy()
                assert data.max() <= 1, 'Please check the file dataset.py, NCANDA.bins wants to put attr values into bins from -1 to 1, but the attr value is not normalized to -1 to 1.'
                if attr in ["cin",'tem','ins']:
                    digitized = np.digitize(data, biased_bins)
                else:
                    digitized = np.digitize(data, bins)
                self.bins[attr] = [data[digitized == i].mean() for i in range(1, len(bins))]   # It's a list with len = 4 (4 bins), each item is the mean value of all values that fall into that bin.
                for i in range(1, len(bins)):
                    if len(data[digitized==i]) == 0:
                        print(attr, i)

    def load_attributes(self, attributes):
        if self.split == 'test' or self.split=='valid':
            csv_path = 'datasets/ncanda/ncanda_test.csv'
        usecols = [att for att in attributes]
        df = pd.read_csv(csv_path, usecols=usecols, header=0)
        
        attribute_dict = {}
        for att in attributes:
            if att == 'diagnosis':
                df[att] = df[att].replace('heavy_with_binging',3)
                df[att] = df[att].replace('heavy',2)
                df[att] = df[att].replace('moderate',1)
                df[att] = df[att].replace('control',0)
            elif att == "sex":
                df[att] = df[att].replace('M',1)
                df[att] = df[att].replace('F',0)
            attribute_dict[att] = df[att].tolist()
        return attribute_dict

    def load_images(self, idx):
        if self.split == 'test' or self.split == 'valid':
            csv_path = 'datasets/ncanda/ncanda_test.csv'

        df = pd.read_csv(csv_path, usecols=['fname'], header=0)
        fnames = df['fname'].tolist()

        data_dir="RAW/MRI_3Set/"
        images = []
        fname = fnames[idx]
        image_path = Path(data_dir + fname)
        img = nib.load(image_path)
        data = img.get_fdata(dtype=np.float32) # (138,176,138)
        max_value = np.percentile(data, 95)
        min_value = np.percentile(data, 5)
        data = np.where(data <= max_value, data, max_value)
        data = np.where(data <= min_value, 0., data)
        data = data / max_value # normalize to [0,1]
        data = data * 2 - 1   # normalize to [-1,1]
        pad_zero = -np.ones((144,176,144), dtype=np.float32)
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
        self.images = torch.tensor(self.load_images(idx), dtype=torch.float32)
        # print(self.images.shape)  # (1, 144, 176, 144)
        
        if self.transform:
            return self.transform(self.images, self.attrs[idx])

        if self.transform_cls:
            return self.transform_cls(self.images), self.attrs[idx]

        return self.images, self.attrs[idx]

