
"""
training_scripts.py
-------------------
Training utilities for deepscm3d models, including vae, hvae, gan.
"""

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
import sys
sys.path.append("../../")
from datasets.transforms import SelectParentAttributesTransform
from models.utils import generate_checkpoint_callback, generate_early_stopping_callback, generate_ema_callback


def get_dataloaders(data_class, attribute_size, config, transform=None, batch_size_overwrite=None,**kwargs):
    data = data_class(attribute_size=attribute_size, transform=transform, split='train', **kwargs)
    if data.has_valid_set:
        train_set = data
        val_set = data_class(attribute_size=attribute_size, transform=transform, split='valid', **kwargs)
    else:
        train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1 - config["train_val_split"]])
    if batch_size_overwrite is not None:
        batch_size_train = batch_size_overwrite
        batch_size_val = batch_size_overwrite
    else:
        batch_size_train = config["batch_size_train"]
        batch_size_val = config["batch_size_val"]
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=7, persistent_workers=True)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_val, shuffle=False, num_workers=7, persistent_workers=True)
    return train_data_loader, val_data_loader

def train_vae(vae, config, data_class, graph_structure, attribute_size, checkpoint_dir, batch_size_overwrite=None, **kwargs):
    transform = SelectParentAttributesTransform("image", attribute_size, graph_structure)

    train_data_loader, val_data_loader = get_dataloaders(data_class, attribute_size, config, transform, batch_size_overwrite, **kwargs)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # To get the batch size and print it in the title of callbacks
    for batch in train_data_loader:
        batch_size = batch[0].shape[0]
        print("Data shape:", batch[0].shape)  # Input data
        print("Labels shape:", batch[1].shape)
        break
    
    callbacks = [
        generate_checkpoint_callback(vae.name, checkpoint_dir, batch_size_log=batch_size),
        generate_early_stopping_callback(patience=config["patience"]),
        lr_monitor
    ]
    if config["ema"] == "True":
        callbacks.append(generate_ema_callback(decay=0.999))
    if vae.name == 'image_hvae':
        strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'auto'
    trainer = Trainer(accelerator="auto", devices="auto", strategy=strategy,
                    callbacks=callbacks,
                    default_root_dir=checkpoint_dir, max_epochs=config["max_epochs"], enable_progress_bar=True)

    trainer.fit(vae, train_data_loader, val_data_loader)

def train_gan(gan, config, data_class, graph_structure, attribute_size, checkpoint_dir, **kwargs):
    transform = SelectParentAttributesTransform("image", attribute_size, graph_structure)

    train_data_loader, val_data_loader = get_dataloaders(data_class, attribute_size, config, transform, **kwargs)

    monitor = 'val_loss'
    for batch in train_data_loader:
        batch_size = batch[0].shape[0]
        print("Data shape:", batch[0].shape)  # Input data
        print("Labels shape:", batch[1].shape)
        break
    callbacks = [
        generate_checkpoint_callback(gan.name, checkpoint_dir, monitor=monitor, batch_size_log=batch_size, top=-1, every_n_epochs=0),
        generate_early_stopping_callback(patience=config["patience"], monitor=monitor)
    ]

    trainer = Trainer(accelerator="auto", devices="auto", strategy="ddp_find_unused_parameters_true",
                      callbacks=callbacks,
                      default_root_dir=checkpoint_dir, max_epochs=config["max_epochs"])

    trainer.fit(gan, train_data_loader, val_data_loader)
