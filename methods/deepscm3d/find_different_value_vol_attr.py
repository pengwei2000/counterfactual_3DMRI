import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

import sys
sys.path.append("../../")

from datasets.ncanda.dataset import NCANDA
from datasets.transforms import ReturnDictTransform
from datasets.ncanda.dataset import unnormalize as unnormalize_ncanda


def different_value(possible_values, value, bins, attribute):
    if bins is not None and attribute in bins:  # age and other vols
        return np.digitize(possible_values, bins[attribute]) != np.searchsorted(bins[attribute], value)
    else:     # sex and diagnosis
        return possible_values != value


def produce_counterfactuals(factual_batch: torch.Tensor, scm: nn.Module, do_parent:str, possible_values = None, device: str = 'cuda', bins = None):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()} 

    possible_values = possible_values[do_parent]
    values = factual_batch[do_parent].cpu()

    interventions = {do_parent: torch.cat([torch.tensor(np.random.choice(possible_values[different_value(possible_values, value, bins, do_parent)])).unsqueeze(0)
                                        for value in values]).view(-1).unsqueeze(1).to(device)}   # (n,1)

    abducted_noise = scm.encode(**factual_batch)
    counterfactual_batch = scm.decode(interventions, **abducted_noise)

    return counterfactual_batch

def find_different_value_vol(test_set: Dataset, batch_size:int=460):
    '''
    Find and save different vol values of test set.
    For each sample in test set, it has its original attr value.
    Here find a different value for each vol attr, and save them.
    So that it can be used later to produce the counterfactual images.
    '''
    metadata = torch.load("ncanda_test_metadata")

    interventions = {}
    attr = ['fro','par','tem','occ','cin','ins','ven']
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)
    result = {}
    for i in range(len(attr)):
        do_parent = attr[i]
        print(f'pa is {do_parent}')
        buffer = np.ones((582, 7))
        for idx, batch in enumerate(test_data_loader):
            for k in range(7):
                buffer[idx*batch_size:(idx+1)*batch_size,k] = unnormalize_ncanda(batch[attr[k]].squeeze(-1).cpu(),attr[k])
            values = batch[do_parent].cpu()
            # print(values)
            possible_values = test_set.possible_values[do_parent]
            bins = test_set.bins
            interventions = torch.cat([torch.tensor(np.random.choice(possible_values[different_value(possible_values, value, bins, do_parent)])).unsqueeze(0)
                                                    for value in values]).view(-1)   # (bz,1)
            # print(interventions)
            buffer[idx*batch_size:(idx+1)*batch_size,i] = unnormalize_ncanda(interventions,do_parent)
        result[do_parent] = buffer
        print('============ this is after change ============')
        print(buffer[:4,:])
        print(buffer[-4:,:])
        print('============ this is before change ============')
        print(metadata[:4,:])
        print(metadata[-4:,:])
    np.savez_compressed('ncanda_target_metadata.npz', **result)


def find_metadata(test_set: Dataset, batch_size:int=582):
    '''
    Find and save different vol values of test set.
    For each sample in test set, it has its original attr value.
    Here find a different value for each vol attr, and save them.
    So that it can be used later to produce the counterfactual images.
    '''

    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)
    for idx, batch in enumerate(test_data_loader):
        values = batch[1].cpu()
        print(values.shape)
        values = values[:,-7:]
        print(values.shape)
    torch.save(values, 'ncanda_test_metadata')

attribute_size = {
        "age": 1,
        "sex": 2,
        "diagnosis": 4,
        "fro": 1,
        "par": 1,
        "tem": 1,
        "occ": 1,
        "cin": 1,
        "ins": 1,
        "ven": 1
    }
transform = ReturnDictTransform(attribute_size)
unnormalize_fn = unnormalize_ncanda
test_set = NCANDA(attribute_size, split='test')
find_metadata(test_set, batch_size=582)
test_set = NCANDA(attribute_size, split='test', transform=transform)
find_different_value_vol(test_set, batch_size=582)