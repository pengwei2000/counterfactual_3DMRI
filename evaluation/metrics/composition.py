
"""
composition.py
----------------
Implements the composition metric for counterfactual image evaluation in 3D MRI benchmarks.

Functions:
    - composition: Computes composition scores and returns image slices for visualization.
    - l1_distance: Computes L1, SSIM, or LPIPS distances between image cycles.
"""

import numpy as np
from evaluation.metrics.ssim import SSIM


def composition(factual_batch, unnormalize_fn, method, cycles=[1, 10], device='cuda', embedding=None, embedding_fn=None):
    """
    Computes the composition metric for a batch of images using a given method.

    Args:
        factual_batch (dict): Batch of input data, must include 'image'.
        unnormalize_fn (callable): Function to unnormalize images for visualization.
        method (object): Model with encode/decode methods.
        cycles (list): List of cycle steps to evaluate.
        device (str): Device to run computation on.
        embedding (str or None): Distance metric ('lpips', 'ssim', or None for L1).
        embedding_fn (callable or None): Embedding function if required.

    Returns:
        tuple: (composition_scores, image_batch_t, image_batch_c, image_batch_s)
    """
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}
    images = [factual_batch["image"]]
    for _ in range(max(cycles)):
        abducted_noise = method.encode(**factual_batch)
        counterfactual_batch = method.decode(**abducted_noise)
        # Replace NaNs with -1 for visualization safety
        counterfactual_batch["image"][counterfactual_batch["image"].isnan()] = -1.
        images.append(counterfactual_batch["image"])
        factual_batch = counterfactual_batch
    composition_scores = l1_distance(images, steps=cycles, embedding=embedding, embedding_fn=embedding_fn)
    t_idx = images[0].shape[-1] // 2
    c_idx = images[0].shape[-2] // 2
    s_idx = images[0].shape[-3] // 2
    # Stack images for all cycles for visualization
    image_batch_t = np.concatenate([
        unnormalize_fn(image[:, :, :, :, t_idx], "image").cpu().numpy() for image in images
    ], axis=3)
    image_batch_c = np.concatenate([
        unnormalize_fn(image[:, :, :, c_idx, :], "image").cpu().numpy() for image in images
    ], axis=3)
    image_batch_s = np.concatenate([
        unnormalize_fn(image[:, :, s_idx, :, :], "image").cpu().numpy() for image in images
    ], axis=3)
    return composition_scores, image_batch_t, image_batch_c, image_batch_s


def l1_distance(images, steps, embedding, embedding_fn):
    """
    Computes distances between images[step] and images[0] for each step.

    Args:
        images (list): List of image tensors for each cycle.
        steps (list): Steps to compute distances for.
        embedding (str or None): Distance metric ('lpips', 'ssim', or None for L1).
        embedding_fn (callable or None): Embedding function if required.

    Returns:
        dict: step -> distance array
    """
    distances = {}
    for step in steps:
        if embedding == "lpips":
            distances[step] = np.array([embedding_fn(images[step], images[0])])
        elif embedding == 'ssim':
            ssim_caller = SSIM(data_dim=3, keep_batch_dim=True).cuda()
            distances[step] = ssim_caller((images[step]+1)/2, (images[0]+1)/2).cpu().numpy()
        elif embedding is None:
            # Ensure no NaNs in the images
            if np.any(np.isnan(np.array(images[step].cpu()))):
                raise ValueError(f'NaN detected in counterfactuals at step {step}.')
            distances[step] = np.mean(
                np.abs(images[step].cpu().numpy() - images[0].cpu().numpy()), axis=(1,2,3,4)
            )  # shape: (batch,)
    return distances
