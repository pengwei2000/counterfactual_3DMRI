import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import nibabel as nib
import torch.nn.functional as F


MIN_MAX = {
    'image': [-1.0, 1.0],
    'age': [55.25, 96.],
    'fro': [136061.0, 212592.0],
    'par':[88797.0, 139170.0],
    'tem':[94958.0, 175299.0],
    'occ':[36153.0, 70856.0],
    'cin':[20419.0, 34470.0],
    'ins':[12401.0, 22950.0],
    'ven':[9527.0, 157058.0]
}

def ordinal_array(num, m=None, reverse=False, scale=1):
    if reverse:
        return scale * torch.count_nonzero(num, dim=1).to(num.device)
    else:
        return np.pad(np.ones(num), (m - num, 0), 'constant').astype(np.float32)

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
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value

class ADNI3d(Dataset):
    def __init__(self, attribute_size, split='train', normalize_=True, transform=None, transform_cls=None):
        super().__init__()
        self.split = split
        self.transform = transform
        self.transform_cls = transform_cls
        self.has_valid_set = True
        self.test_idx = np.load('datasets/adni_3d/test_idx.npy').tolist()
        self.train_idx = np.load('datasets/adni_3d/train_idx.npy').tolist()
        assert len(self.test_idx)+len(self.train_idx)==4578, f'The train and test set should be 4578'
        
        self.attributes = self.load_attributes(attributes=attribute_size.keys())
        
        if normalize_:
            self.attributes = {attr: normalize(torch.tensor(values, dtype=torch.float32), attr) for attr, values in self.attributes.items()}
        else:
            self.attributes = {attr: torch.tensor(values, dtype=torch.float32) for attr, values in self.attributes.items()}
        
        # change to one-hot representation
        if 'sex' in self.attributes:
            self.attributes['sex'] = F.one_hot(self.attributes['sex'].to(torch.long), num_classes=2)
        if 'diagnosis' in self.attributes:
            self.attributes['diagnosis'] = F.one_hot(self.attributes['diagnosis'].to(torch.long), num_classes=5)


        self.attrs = torch.cat([self.attributes[attr].unsqueeze(1) if len(self.attributes[attr].shape) == 1 else self.attributes[attr]
                                for attr in attribute_size.keys()], dim=1)
        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.attributes.items()}

        bins = np.array([-1, 0 ,1])
        biased_bins = np.array([-1, -0.35, 1])

        self.bins = {}
        for attr, values in self.attributes.items():
            if attr not in ["sex", "diagnosis"]:
                data = values.numpy()
                assert data.max() <= 1, 'Please check the file dataset.py, ADNI3d.bins wants to put attr values into bins from -1 to 1, but the attr value is not normalized to -1 to 1.'
                if attr in ["cin",'tem','ins']:
                    digitized = np.digitize(data, biased_bins)
                else:
                    digitized = np.digitize(data, bins)
                self.bins[attr] = [data[digitized == i].mean() for i in range(1, len(bins))]   # It's a list with len = 4 (4 bins), each item is the mean value of all values that fall into that bin.
                for i in range(1, len(bins)):
                    if len(data[digitized==i]) == 0:
                        print(attr, i)

    def load_attributes(self, attributes):
        csv_path = 'datasets/adni_3d/adni_3d_ps.csv'
        usecols = [att for att in attributes]
        df = pd.read_csv(csv_path, usecols=usecols, header=0)
        
        if self.split == 'train':
            df = df.loc[self.train_idx]
        elif self.split == 'valid' or self.split == "test":
            df = df.loc[self.test_idx]
        attribute_dict = {}
        for att in attributes:
            if att == 'diagnosis':
                df[att] = df[att].replace('AD',4)
                df[att] = df[att].replace('LMCI',3)
                df[att] = df[att].replace('EMCI',2)
                df[att] = df[att].replace('SMC',1)
                df[att] = df[att].replace('CN',0)
            elif att == "sex":
                df[att] = df[att].replace('M',1)
                df[att] = df[att].replace('F',0)
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
    