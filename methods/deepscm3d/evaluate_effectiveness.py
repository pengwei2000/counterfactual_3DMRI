import torch
import numpy as np
from typing import List
from json import load
from importlib import import_module
from counterfactual_benchmark.methods.deepscm3d.model import SCM
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import argparse
import sys
sys.path.append("../../")

from datasets.adni_3d.dataset import ADNI3d
from datasets.transforms import ReturnDictTransform
from datasets.adni_3d.dataset import unnormalize as unnormalize_adni3d

torch.multiprocessing.set_sharing_strategy('file_system')
rng = np.random.default_rng()

dataclass_mapping = {
    "adni3d": (ADNI3d, unnormalize_adni3d)
}

def different_value(possible_values, value, bins, attribute):
    if bins is not None and attribute in bins:
        return np.digitize(possible_values, bins[attribute]) != np.searchsorted(bins[attribute], value)
    else:
        return possible_values != value

def produce_counterfactuals(factual_batch: torch.Tensor, scm: nn.Module, do_parent:str, device: str = 'cuda', possible_values = None, bins = None):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}   

    possible_values = possible_values[do_parent]
    values = factual_batch[do_parent].cpu()
    batch_size = factual_batch[do_parent].shape[0]

    interventions = {do_parent: torch.cat([torch.tensor(np.random.choice(possible_values[different_value(possible_values, value, bins, do_parent)])).unsqueeze(0)
                                        for value in values]).view(-1).unsqueeze(1).to(device)}   # (n,1)

    abducted_noise = scm.encode(**factual_batch)
    counterfactual_batch = scm.decode(interventions, **abducted_noise)
    counterfactual_metadata = np.ones((batch_size,7))
    attrs = ['fro','par','tem','occ','cin','ins','ven']
    for j in range(len(attrs)):
        counterfactual_metadata[:,j] = unnormalize_adni3d(counterfactual_batch[attrs[j]], attrs[j])
    return counterfactual_batch, counterfactual_metadata

@torch.no_grad()
def evaluate_effectiveness(test_set: Dataset, unnormalize_fn, batch_size:int , scm: nn.Module, attributes: List[str], do_parent:str, model='gan'):

    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7, persistent_workers=True)
    idx = 0

    directory = f"/SynthSeg/evaluation/{do_parent}/{model}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    metadata_all = np.ones((460,7))
    for factual_batch in tqdm(test_data_loader):
        counterfactuals, metadata = produce_counterfactuals(factual_batch, scm, do_parent, possible_values=test_set.possible_values, bins=test_set.bins)
        images = counterfactuals['image']
        for i in range(images.shape[0]):
            file_path = os.path.join(directory, f"counterfactual_{do_parent}_{str(idx).zfill(3)}.npz")
            vol_data = images[i].cpu().squeeze()
            assert vol_data.shape == (144,176,144)
            np.savez_compressed(file_path, vol_data=vol_data)
            metadata_all[idx] = metadata[i]
            idx += 1
    print(f"{idx} counterfactuals of {do_parent} saved successfully.")
    np.save(f'{model}-doparent-{do_parent}-metadata.npy', metadata_all)
    return

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, help="Config file for experiment.", default="./configs/adni3d/vae.json")
    parser.add_argument("--model", type=str)
    parser.add_argument("--metrics", '-m',
                        nargs="+", type=str,
                        help="Metrics to calculate. "
                        "Choose one or more of [composition, effectiveness, fid, minimality]. If not set, all metrics are calculated.",
                        default=["effectiveness"])
    parser.add_argument("--sampling-temperature", '-temp', type=float, default=0.1, help="Sampling temperature, used for VAE, HVAE.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    # torch.manual_seed(42)

    assert os.path.isfile(args.config), f"{args.config} is not a file"
    with open(args.config, 'r') as f:
        config = load(f)

    dataset = config["dataset"]
    attribute_size = config["attribute_size"]

    models = {}
    for variable in config["causal_graph"].keys():
        if variable not in config["mechanism_models"]:
            continue
        model_config = config["mechanism_models"][variable]

        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])
        model = model_class(params=model_config["params"], attr_size=attribute_size)

        models[variable] = model
        if "finetune" in model_config["params"] and model_config["params"]["finetune"] == 1:
            model.name += '_finetuned'

    batch_size = config["mechanism_models"]["image"]["params"]["batch_size_val"]
    
    scm = SCM(checkpoint_dir=config["checkpoint_dir"],
              graph_structure=config["causal_graph"],
              temperature=args.sampling_temperature,
              **models)

    data_class, unnormalize_fn = dataclass_mapping[dataset]
    transform = ReturnDictTransform(attribute_size)
    test_set = data_class(attribute_size, split='test', transform=transform)
    attribute_list = ['fro','par','tem','occ','cin','ins','ven']

    arrays = {}
    for i in range(len(attribute_list)):
        pa = attribute_list[i]
        print(f'saving counterfactuals for intervention on: {pa}')
        evaluate_effectiveness(test_set, unnormalize_fn, batch_size, scm=scm, attributes=attribute_list, do_parent=pa, model=args.model)  # (460,7)