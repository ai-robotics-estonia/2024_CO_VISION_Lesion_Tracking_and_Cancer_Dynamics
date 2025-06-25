from cgi import test
import os
import json
import hydra
import wandb
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import psutil

import torch
import torch.nn.functional as F
from monai.networks.blocks import Warp
from omegaconf import OmegaConf
from pathlib import Path

from losses.registration_loss import TRE
from einops import rearrange

from models.vox2vox import GeneratorUNetGlobal, Discriminator
from losses.vox2vox_losses import VesselLoss, LNCC

class Vox2VoxReg(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hparams.update(config)
        self.save_hyperparameters()

        # Just a workaround to properly save hyperparameters to WandB but still use as OmegaConf
        self.hparams.update(OmegaConf.create(config))

        self.process = psutil.Process()
        self.cpus = list(range(psutil.cpu_count()))
        self.process.cpu_affinity(self.cpus)

        self.generator = GeneratorUNetGlobal()
        self.discriminator = Discriminator()
        self.warp_layer = Warp()

        self.dice_metric_before = hydra.utils.instantiate(self.hparams.metric)
        self.dice_metric_after = hydra.utils.instantiate(self.hparams.metric)

        self.vx = torch.Tensor(self.hparams.datamodule.config.vx).to('cuda:0')
        self.tre = TRE(vx=self.vx)

        self.lncc = LNCC()
        self.vessel_loss = VesselLoss()
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss().to('cuda') 

        self.val_tre_before = []
        self.val_tre_after = []


    def forward(self, real_A, real_B):
        return self.generator(torch.cat([real_B, real_A], dim=1))
    


    def training_step(self, batch, batch_idx, optimizer_idx):
        fixed, moving = batch["fixed_image"], batch["moving_image"]
        lung_A, lung_B = batch["fixed_label"], batch["moving_label"]
        vessels_A, vessels_B = batch["fixed_vessels"], batch["moving_vessels"]

        real_A = fixed.requires_grad_(True)
        real_B = moving.requires_grad_(True)
        patch_size = real_A.shape[-3] // 2 ** 4
        valid = torch.ones((real_A.size(0), patch_size, patch_size, patch_size), device=self.device)
        fake = torch.zeros_like(valid)