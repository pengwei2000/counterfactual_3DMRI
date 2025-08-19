import sys
sys.path.append("../../")

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from datasets.adni_3d.dataset import bin_array


def save_image(img, path):
    if img.shape[0] == 3:
        plt.imsave(path, img.transpose(1, 2, 0))
    else:
        plt.imsave(path, img[0], cmap='gray')
    return

def save_selected_images(images, scores, save_dir, lower_better=True, n_best=5, n_worst=5, n_median=5):
    sort_ids = np.argsort(scores)
    images_sorted = images[sort_ids] if lower_better else images[sort_ids][::-1]

    for i in range(n_best):
        save_image(images_sorted[i], os.path.join(save_dir, f"best_{i}.png"))

    for i in range(n_worst):
        save_image(images_sorted[-i - 1], os.path.join(save_dir, f"worst_{i}.png"))

    total = scores.shape[0]
    for i in range(n_median):
        save_image(images_sorted[(total//2) - (n_median//2) + i], os.path.join(save_dir, f"median_{i}.png"))

    return

def save_selected_images_3d(images_t, images_c, images_s, scores, save_dir, n_best=5, n_worst=5, n_median=5):
    sort_ids = np.argsort(scores)
    # for i in range(n_best):
    #     save_image_3d(images_t[sort_ids][i], images_c[sort_ids][i], images_s[sort_ids][i], os.path.join(save_dir, f"best_{i}.png"))

    # for i in range(n_worst):
    #     save_image_3d(images_t[sort_ids][-i - 1],images_c[sort_ids][-i - 1],images_s[sort_ids][-i - 1], os.path.join(save_dir, f"worst_{i}.png"))

    # total = scores.shape[0]
    # for i in range(n_median):
    #     save_image_3d(images_t[sort_ids][(total//2) - (n_median//2) + i],images_c[sort_ids][(total//2) - (n_median//2) + i],images_s[sort_ids][(total//2) - (n_median//2) + i], os.path.join(save_dir, f"median_{i}.png"))
    
    for i in range(1, 100, 10):
        save_image_3d(images_t[i],images_c[i],images_s[i], os.path.join(save_dir, f"selected_testset_idx_{i}.png"))
    
    return

def save_image_3d(img_t, img_c, img_s, path):
    fig, axs = plt.subplots(3,1, figsize=(20, 10))
    axs[0].imshow(img_t[0], cmap='gray')
    axs[1].imshow(img_c[0], cmap='gray')
    axs[2].imshow(img_s[0], cmap='gray')
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(path)
    return

def to_value(tensor, name, unnormalize_fn):
    if name in ['Smiling', 'Eyeglasses']:
        return "True" if tensor.item() == 1.0 else "False"
    elif name == 'sex':
        return 'Female' if torch.argmax(tensor, dim=1).item() == 0. else 'Male'
    elif name in ['age', 'fro_vol', 'ins_vol', "other_vol", 'par_vol']:
        unnormalized = unnormalize_fn(tensor.item(), name)
        return round(unnormalized) if name == 'age' else f'{round((unnormalized/1000), 2)}'
    elif name == 'diagnosis':
        diagnosis_list = ["CN", "SMC", "EMCI", "LMCI", "AD"]
        return diagnosis_list[int(torch.argmax(tensor, dim=1).item())]
    elif name == 'apoE':
        return int(bin_array(tensor, reverse=True))
    elif name == "digit":
        return torch.argmax(tensor, dim=1).item()
    elif name == "depth":
        return torch.argmax(tensor, dim=1).item() + 2
    else:
        return round(tensor.item(), 2)
