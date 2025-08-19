"""
embeddings.py
-------------
Utility functions for loading and using LPIPS embedding model for 3D MRI counterfactual benchmarks.

"""

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import sys
from functools import partial
sys.path.append("../../")
from models.utils import rgbify
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC


def get_embedding_model(embedding, pretrained_vgg, classifier_config=None):
    if embedding == "lpips":
        return LPIPS(net_type='vgg', normalize=True).to('cuda')
    else:
        return None

def get_embedding_fn(embedding, unnormalize_fn, embedding_model):
    if embedding is None:
        return partial(unnormalize_embedding_fn, unnormalize_fn)
    elif embedding == "lpips":
        return partial(lpips_embedding_fn, embedding_model)
    else:
        exit(f"Invalid embedding: {embedding}")

def unnormalize_embedding_fn(unnormalize_fn, x, _):
    return unnormalize_fn(x, "image").cpu().numpy()

def lpips_embedding_fn(embedding_model, x, y):
    return embedding_model(rgbify(x, normalized=True), rgbify(y, normalized=True)).detach().cpu().numpy()
