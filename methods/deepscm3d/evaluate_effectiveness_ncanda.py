import torch
import numpy as np
from typing import List
from json import load
from importlib import import_module
from model import SCM
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import argparse
import sys
sys.path.append("../../")

from datasets.ncanda.dataset import NCANDA
from datasets.transforms import ReturnDictTransform
from datasets.ncanda.dataset import unnormalize as unnormalize_ncanda
from datasets.ncanda.dataset import normalize as normalize_ncanda

torch.multiprocessing.set_sharing_strategy('file_system')

dataclass_mapping = {
    "ncanda": (NCANDA, unnormalize_ncanda)
}

def produce_counterfactuals(factual_batch: torch.Tensor, scm: nn.Module, do_parent:str, device: str = 'cuda', intervention = None):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}

    interventions = {do_parent: intervention.to(device)}   # (n,1)

    abducted_noise = scm.encode(**factual_batch)
    counterfactual_batch = scm.decode(interventions, **abducted_noise)

    return counterfactual_batch

@torch.no_grad()
def evaluate_effectiveness(test_set: Dataset, unnormalize_fn, batch_size:int , scm: nn.Module, attributes: List[str], do_parent:str, intervention=None, model=None):

    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7, persistent_workers=True)
    idx = 0

    directory = f"/SynthSeg/evaluation/ncanda/{model}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for k, factual_batch in enumerate(tqdm(test_data_loader)):
        intervention_batch = intervention[k*batch_size:(k+1)*batch_size,:]

        assert intervention_batch.shape == (batch_size,1)
        counterfactuals = produce_counterfactuals(factual_batch, scm, do_parent, intervention=intervention_batch)
        images = counterfactuals['image']
        for i in range(images.shape[0]):

            file_path = os.path.join(directory, f"counterfactual_{do_parent}_{str(idx).zfill(3)}.npz")
            vol_data = images[i].cpu().squeeze()
            assert vol_data.shape == (144,176,144)
            np.savez_compressed(file_path, vol_data=vol_data)
            idx += 1
    print(f"{idx} counterfactuals of {do_parent} saved successfully.")
    return

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, help="Config file for experiment.", default="./configs/ncanda/gan.json")
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


    unnormalize_fn = unnormalize_ncanda
    transform = ReturnDictTransform(attribute_size)
    test_set = NCANDA(attribute_size, split='test', transform=transform)
    attribute_list = ['fro','par','tem','occ','cin','ins','ven']
    arrays = {}
    if args.model is None:
        raise ValueError
    intervention_source = np.load('counterfactual-benchmark/counterfactual_benchmark/methods/deepscm3d/ncanda_target_metadata.npz')
    for i in range(len(attribute_list)):
        pa = attribute_list[i]
        print(f'saving counterfactuals for intervention on: {pa}')
        intervention = normalize_ncanda(torch.from_numpy(intervention_source[pa][:,i][:,None]),name=pa).float()
        assert intervention.shape == (582,1)
        evaluate_effectiveness(test_set, unnormalize_fn, batch_size, scm=scm, attributes=attribute_list, do_parent=pa, model=args.model, intervention=intervention)  # (460,7)


