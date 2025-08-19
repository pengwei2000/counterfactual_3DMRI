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
import random

import sys
sys.path.append("../../")

from datasets.adni_3d.dataset import ADNI3d
from datasets.transforms import ReturnDictTransform

from evaluation.metrics.composition import composition
from evaluation.metrics.ssim import SSIM
from evaluation.metrics.utils import save_plots, save_selected_images_3d
from datasets.adni_3d.dataset import unnormalize as unnormalize_adni3d

torch.multiprocessing.set_sharing_strategy('file_system')

dataclass_mapping = {
    "adni3d": (ADNI3d, unnormalize_adni3d)
}

def produce_qualitative_samples(dataset, scm, parents, unnormalize_fn, num=20, show_difference=False):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=7)

    produce_every = len(dataset) // num
    fig_idx = 0
    for i , batch in tqdm(enumerate(data_loader)):
        if i % produce_every == 0:
            res = [batch]

            for do_parent in parents:
                counterfactual = produce_counterfactuals(batch, scm, do_parent, possible_values=dataset.possible_values)
                res.append(counterfactual)

            save_plots(res, fig_idx, parents, unnormalize_fn, show_difference=show_difference)
            fig_idx += 1
    return


def evaluate_composition(test_set: Dataset, unnormalize_fn, batch_size: int, cycles: List[int], scm: nn.Module, save_dir: str = "composition_samples", embedding = None, embedding_fn = None):
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)

    composition_scores = []

    images_t = []
    images_c = []
    images_s = []
    for factual_batch in tqdm(test_data_loader):
        score_batch, image_batch_t, image_batch_c, image_batch_s = composition(factual_batch, unnormalize_fn, method=scm, cycles=cycles, embedding=embedding, embedding_fn=embedding_fn)
        composition_scores.append(score_batch)
        
        images_t.append(image_batch_t)
        images_c.append(image_batch_c)
        images_s.append(image_batch_s)

    images_t = np.concatenate(images_t) # (allbatch, 1, 144, 176 * (max(cycle)+1))
    images_c = np.concatenate(images_c)
    images_s = np.concatenate(images_s)

    composition_scores = {cycle: np.concatenate([composition_batch[cycle] for composition_batch in composition_scores]) for cycle in cycles}
    
    os.makedirs(save_dir, exist_ok=True)
    save_selected_images_3d(images_t, images_c, images_s, composition_scores[cycles[-1]], save_dir=save_dir)

    for cycle in cycles:
        print(f"Average composition score for {cycle} cycles: mean {round(np.mean(composition_scores[cycle]), 3):.3f} std {round(np.std(composition_scores[cycle]), 3):.3f}")
    return

def evaluate_reversibility(test_set: Dataset, unnormalize_fn, batch_size: int, scm: nn.Module, embedding: None, do_parent: str, save_dir: str = "reversibility_samples/gan/", cycles= [1,3]):
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)
    reversibility_scores = []
    images_t = []
    images_c = []
    images_s = []

    for factual_batch in tqdm(test_data_loader):
        image_batch = [factual_batch['image']]
        for i in range(max(cycles)):
            if i == 0:
                counterfactuals, orig_pa, interv_pa = produce_counterfactuals_reversibility(factual_batch, scm, do_parent, possible_values=test_set.possible_values, bins=test_set.bins, cycle=i)
            else:
                counterfactuals = produce_counterfactuals_reversibility(counterfactuals, scm, do_parent, original_parent_value=orig_pa, intervention_parent_value=interv_pa, cycle=i)
            image_batch.append(counterfactuals["image"])

        score = l1_distance_3d(image_batch, steps=cycles, embedding=embedding)   # dict, key=step/cycle number, value = (bz,)
        reversibility_scores.append(score)
        # stack images for all cycles
        t_idx = image_batch[0].shape[-1] // 2
        c_idx = image_batch[0].shape[-2] // 2
        s_idx = image_batch[0].shape[-3] // 2
        image_batch_t = np.concatenate([unnormalize_fn(image[:,:,:,:,t_idx], "image").cpu().numpy() for image in image_batch], axis=3)  # (batch, 1, 144, 176 * (max(cycles)+1) )  
        image_batch_c = np.concatenate([unnormalize_fn(image[:,:,:,c_idx,:], "image").cpu().numpy() for image in image_batch], axis=3) # (batch, 1, 144, 144 * (max(cycles)+1) )  
        image_batch_s = np.concatenate([unnormalize_fn(image[:,:,s_idx,:,:], "image").cpu().numpy() for image in image_batch], axis=3) # (batch, 1, 144, 144 * (max(cycles)+1) )  
        images_t.append(image_batch_t)
        images_c.append(image_batch_c)
        images_s.append(image_batch_s)
    images_t = np.concatenate(images_t) # (allbatch, 1, 144, 176 * (max(cycle)+1))
    images_c = np.concatenate(images_c)
    images_s = np.concatenate(images_s)
    reversibility_scores = {cycle: np.concatenate([reversibility_batch[cycle] for reversibility_batch in reversibility_scores]) for cycle in cycles}
    for cycle in cycles:
        print(f"Average reversibility score for do_parent {do_parent} for {cycle} cycles: mean {round(np.mean(reversibility_scores[cycle]), 3):.3f} std {round(np.std(reversibility_scores[cycle]), 3):.3f}")
    
    save_dir += do_parent
    os.makedirs(save_dir, exist_ok=True)
    save_selected_images_3d(images_t, images_c, images_s, reversibility_scores[cycles[-1]], save_dir=save_dir)
    return

def l1_distance_3d(image_batch, steps, embedding: None):
    distances = {}
    for step in steps:
        if embedding == 'ssim':
            ssim_caller = SSIM(data_dim=3, keep_batch_dim=True).cuda()
            distances[step] = ssim_caller((images[step]+1)/2,(images[0]+1)/2)
        elif embedding is None:
            distances[step] = np.mean(np.abs(image_batch[step].cpu().numpy() - image_batch[0].cpu().numpy()), axis=(1,2,3,4))        # shape: (batch,)
    return distances

def different_value(possible_values, value, bins, attribute):
    if bins is not None and attribute in bins:
        return np.digitize(possible_values, bins[attribute]) != np.searchsorted(bins[attribute], value)
    else:
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


def produce_counterfactuals_reversibility(factual_batch: torch.Tensor, scm: nn.Module, do_parent:str, possible_values = None, device: str = 'cuda', bins = None, original_parent_value=None, intervention_parent_value=None, cycle:int=0):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}
    if cycle == 0:  # The first pass from original batch to intervention batch, and from intervention to original batch
        possible_values = possible_values[do_parent]
        values = factual_batch[do_parent].cpu()
        original_parent_value = values.reshape(values.shape[0],-1)

        interventions = {do_parent: torch.cat([torch.tensor(np.random.choice(possible_values[different_value(possible_values, value, bins, do_parent)])).unsqueeze(0)
                                            for value in values]).view(-1).unsqueeze(1).to(device)}   # (n,1)

        intervention_parent_value = interventions[do_parent]
        abducted_noise = scm.encode(**factual_batch)
        counterfactual_batch = scm.decode(interventions, **abducted_noise) # from orignal to intervention

        interventions = {do_parent: original_parent_value.to(device)}
        abducted_noise = scm.encode(**counterfactual_batch)
        counterfactual_batch = scm.decode(interventions, **abducted_noise)  # from intervention to orignal
        
        assert original_parent_value.shape == intervention_parent_value.shape
        return counterfactual_batch, original_parent_value, intervention_parent_value
    else:
        interventions = {do_parent: intervention_parent_value.to(device)}

        abducted_noise = scm.encode(**factual_batch)
        counterfactual_batch = scm.decode(interventions, **abducted_noise)  # from orignal to intervention
        interventions = {do_parent: original_parent_value.to(device)}
        abducted_noise = scm.encode(**counterfactual_batch)
        counterfactual_batch = scm.decode(interventions, **abducted_noise)  # from intervention to orignal
        return counterfactual_batch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, help="Config file for experiment.", default="./configs/adni3d/vae.json")
    parser.add_argument("--metrics", '-m',
                        nargs="+", type=str,
                        help="Metrics to calculate. "
                        "Choose one or more of [composition, reversibility]. If not set, all metrics are calculated.",
                        choices=["composition", "reversibility"],
                        default=["composition", 'reversibility'])
    parser.add_argument("--cycles", '-cc', nargs="+", type=int, help="Composition cycles.", default=[1, 10])
    parser.add_argument("--qualitative", '-qn', type=int, help="Number of qualitative results to produce", default=0)
    parser.add_argument("--show-difference", '-sd', action='store_true', help="Show counterfactual-factual difference on qualitative results")
    parser.add_argument("--embeddings", type=str, choices=["ssim"], help="What embeddings to use for composition and reversibility metric. "
                        "Supported: [ssim]. If not set, will compute distance on image space")
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

    data_class, unnormalize_fn = dataclass_mapping[dataset]

    transform = ReturnDictTransform(attribute_size)
    train_set = data_class(attribute_size, split='train', transform=transform)
    test_set = data_class(attribute_size, split='test', transform=transform)

    if args.qualitative > 0:
        produce_qualitative_samples(dataset=test_set, scm=scm, parents=list(attribute_size.keys()), unnormalize_fn=unnormalize_fn, num=args.qualitative,
                                    show_difference=args.show_difference)

    if "composition" in args.metrics:
        evaluate_composition(test_set, unnormalize_fn, batch_size, cycles=args.cycles, scm=scm, embedding=args.embeddings)
    
    if "reversibility" in args.metrics:
        for pa in ['fro','par','tem','occ','cin','ins','ven','age','sex']:
            evaluate_reversibility(test_set, unnormalize_fn, batch_size, scm=scm, do_parent=pa, embedding=args.embeddings)
