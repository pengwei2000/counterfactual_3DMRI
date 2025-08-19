"""
utils.py
--------
Utility functions for model training, checkpointing, configuration, and tensor operations in 3D MRI counterfactual benchmarks.
"""

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from json import load
import argparse
import torch.nn as nn
import torch
import os
from .weight_averaging import EMA

def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Set requires_grad=False for all parameters in the model.
    """
    for p in model.parameters():
        p.requires_grad = False
    return model

def generate_checkpoint_callback(model_name, dir_path, monitor="val_loss", mode="min", save_last=False, top=1, batch_size_log=None, every_n_epochs=None):
    checkpoint_callback = ModelCheckpoint(
    dirpath=dir_path,
    filename= model_name + '-{epoch:02d}-{val_loss:.3f}'+f'-batch_size={batch_size_log}',
    monitor=monitor,  # Disable monitoring for checkpoint saving,
    mode = mode,
    save_top_k=top,
    every_n_epochs=every_n_epochs,
    save_last=save_last
    )
    return checkpoint_callback

def generate_checkpoint_callback_gan(model_name, dir_path, monitor="val_loss", save_last=False, every_n_epochs=1, batch_size_log=None):
    checkpoint_callback = ModelCheckpoint(
    dirpath=dir_path,
    every_n_epochs = every_n_epochs,
    filename= model_name + '-{epoch:02d}-{monitor:.3f}'+f'-batch_size={batch_size_log}',
    monitor=monitor,  # Disable monitoring for checkpoint saving,
    save_last=save_last
    )
    return checkpoint_callback

def generate_early_stopping_callback(patience=5, min_delta = 0.001, monitor="val_loss", mode="min"):
    early_stopping_callback = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, mode = mode)
    return early_stopping_callback

def generate_ema_callback(decay=0.999):
    ema_callback=EMA(decay=decay)
    return ema_callback

def flatten_list(list):
     return sum(list, [])

def get_config(config_dir, default):
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of the config file to choose", type=str, default=default)
    args = argParser.parse_args()
    config = load(open(config_dir + args.name + ".json", "r"))
    return config

def override(f):
    return f

def overload(f):
    return f

def linear_warmup(warmup_iters):
    def f(iter):
        return 1.0 if iter > warmup_iters else iter / warmup_iters

    return f

def init_bias(m):
    if type(m) == nn.Conv2d:
        nn.init.zeros_(m.bias)

def init_weights(layer, std=0.01):
    name = layer.__class__.__name__
    if name == 'ConvBlock':
        return
    if name.startswith('Conv'):
        torch.nn.init.normal_(layer.weight, mean=0, std=std)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)

def continuous_feature_map(c: torch.Tensor, size: tuple = (32, 32)):
    return c.reshape((c.size(0), 1, 1, 1)).repeat(1, 1, *size)

def rgbify(image, normalized=True):
    if image.shape[1] == 1:
        if normalized:
            # [-1, 1] -> [0, 1]
            image = (image + 1) / 2
        image = image.repeat(1, 3, 1, 1)  # shape(batch, 3, ...)
    return torch.clamp(image, min=0, max=1)
