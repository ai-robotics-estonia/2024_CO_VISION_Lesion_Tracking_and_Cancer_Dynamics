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


class RegistrationModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hparams.update(config)
        self.save_hyperparameters()

        # Just a workaround to properly save hyperparameters to WandB but still use as OmegaConf
        self.hparams.update(OmegaConf.create(config))

        self.process = psutil.Process()
        self.cpus = list(range(psutil.cpu_count()))
        self.process.cpu_affinity(self.cpus)

        self.net = hydra.utils.instantiate(self.hparams.model)

        self.dice_metric_before = hydra.utils.instantiate(self.hparams.metric)
        self.dice_metric_after = hydra.utils.instantiate(self.hparams.metric)
        self.warp_layer = Warp()

        self.vx = torch.Tensor(self.hparams.datamodule.config.vx).to('cuda:0')
        self.loss = hydra.utils.instantiate(self.hparams.loss)
        self.tre = TRE(vx=self.vx)

        self.val_tre_before = []
        self.val_tre_after = []


    def load_pretrain(self, ckpt_path):
        state_dict = torch.load(ckpt_path)["state_dict"]
        fixed_state_dict = {
            key.replace("net.", ""): value for key, value in state_dict.items()
        }
        self.net.load_state_dict(fixed_state_dict)

    def forward(self, batch_data):
        """
        Model forward pass: predict DDF, warp moving images/labels/keypoints
        """

        fixed_image = batch_data["fixed_image"]
        moving_image = batch_data["moving_image"]
        moving_label = batch_data["moving_label"]
        fixed_label = batch_data["fixed_label"]
        fixed_keypoints = batch_data["fixed_keypoints"]
        moving_keypoints = batch_data["moving_keypoints"]

        batch_size = fixed_image.shape[0]

        # predict DDF through LocalNet
        ddf_image = self.net(torch.cat((moving_image, fixed_image), dim=1)).float()

        # warp moving image and label with the predicted ddf
        pred_image = self.warp_layer(moving_image, ddf_image)

        # warp moving label (optional)
        if moving_label is not None:
            pred_label = self.warp_layer(moving_label, ddf_image)
        else:
            pred_label = None

        # warp vectors for keypoints (optional)
        # figure out how they are calculated
        # grid specifies the sampling pixel locations normalized by the input spatial dimensions. 
        # Therefore, it should have most values in the range of [-1, 1].
        # For example, values x = -1, y = -1 is the left-top pixel of input, 
        # and values x = 1, y = 1 is the right-bottom pixel of input.
        if fixed_keypoints is not None:
            with torch.no_grad():
                # find center of fixed image
                offset = torch.as_tensor(fixed_image.shape[-3:]).to(fixed_keypoints.device) / 2
                offset = offset[None][None]
                # torch.flip(..., (-1,)) flips the coordinates because grid sampling in PyTorch 
                # expects coordinates in the range [-1, 1] in the order (z, y, x).
                # normalize keypoints to [-1, 1] range wrt fixed image center
                ddf_keypoints = torch.flip((fixed_keypoints - offset) / offset, (-1,))
            ddf_keypoints = (
                F.grid_sample(ddf_image, ddf_keypoints.view(batch_size, -1, 1, 1, 3))
                .view(batch_size, 3, -1)
                # [batch_size, num_keypoints, 3]
                .permute((0, 2, 1))
            )
        else:
            ddf_keypoints = None

        return ddf_image, ddf_keypoints, pred_image, pred_label

    def training_step(self, batch):
        ddf_image, ddf_keypoints, pred_image, pred_label = self(batch)

        loss = self.loss(batch['fixed_image'], pred_image,
                         batch['fixed_label'], pred_label,
                         ddf_image,
                         batch['fixed_keypoints'] + ddf_keypoints,
                         batch['moving_keypoints'],
                         )
        tre_before = self.tre(batch['fixed_keypoints'], batch['moving_keypoints'])
        tre_after = self.tre(batch['fixed_keypoints'] + ddf_keypoints, batch['moving_keypoints'])

        train_loss_dict = dict()

        train_loss_dict['train_tre_before'] = tre_before
        train_loss_dict['train_tre_after'] = tre_after
        train_loss_dict['train_loss'] = loss

        # due to using monai dataloader with no collation, batch size cannot be inferred and should be passed manually to the logger
        for loss_name, loss in train_loss_dict.items():
            self.log(
                loss_name,
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                # batch_size=self.batch_size,
            )
        return {
            "loss": train_loss_dict['train_loss'],
        }


    def validation_step(self, batch, batch_idx):

        ddf_image, ddf_keypoints, pred_image, pred_label = self(batch)

        tre_before = self.tre(batch['fixed_keypoints'], batch['moving_keypoints'])
        tre_after = self.tre(batch['fixed_keypoints'] + ddf_keypoints, batch['moving_keypoints'])
        self.val_tre_before.append(tre_before)
        self.val_tre_after.append(tre_after)

        pred_label = pred_label.round()
        self.dice_metric_before(y_pred=batch['moving_label'], y=batch['fixed_label'])
        self.dice_metric_after(y_pred=pred_label, y=batch['fixed_label'])
        

    def on_validation_epoch_end(self):

        # self.process.cpu_affinity(self.cpus)
        
        dice_before = self.dice_metric_before.aggregate().item()
        dice_after = self.dice_metric_after.aggregate().item()

        self.log(
            "val_dice",
            dice_after,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            # batch_size=self.batch_size,
        )

        self.dice_metric_before.reset()
        self.dice_metric_after.reset()

        tre_before = sum(self.val_tre_before) / len(self.val_tre_before)
        tre_after = sum(self.val_tre_after) / len(self.val_tre_after)

        self.log(
            "val_tre",
            tre_after,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            # batch_size=self.batch_size,
        )
        self.val_tre_before = []
        self.val_tre_after = []
    
    # def on_validation_epoch_end(self):
    #      self.process.cpu_affinity(self.cpus)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, self.net.parameters())

        if after_scheduler := self.hparams.scheduling.scheduler.get('after_scheduler', None):
            after_scheduler = hydra.utils.instantiate(after_scheduler, optimizer=optimizer)
            scheduler = hydra.utils.instantiate(
                self.hparams.scheduling.scheduler, optimizer=optimizer, after_scheduler=after_scheduler
        )
        else:
            scheduler = hydra.utils.instantiate(
                self.hparams.scheduling.scheduler, optimizer=optimizer)
        scheduler = {
            "scheduler": scheduler,
            **self.hparams.scheduling.additional_params,
        }
        return [optimizer], [scheduler]
    
    # TODO: fix this (copied from detector model)
    def save_jit_model(self, val_epoch_metric):
        model_name = f"epoch={self.current_epoch}-val_metric={val_epoch_metric:.3f}.pt"

        # remove previously saved last JIT model
        list(map(os.remove, self.ckpt_dirpath.rglob('last-*.pt')))
        torch.jit.save(self.net.network, self.ckpt_dirpath / f"last-{model_name}")

        if val_epoch_metric > self.best_metric:
            self.best_metric = val_epoch_metric
            # remove previously saved best JIT model
            list(map(os.remove, self.ckpt_dirpath.rglob('epoch=*.pt')))
            torch.jit.save(self.net.network, self.ckpt_dirpath / model_name)

    def on_train_epoch_start(self):
        self.process.cpu_affinity(self.cpus)

    def on_validation_epoch_start(self):
        self.process.cpu_affinity(self.cpus)

    def on_test_epoch_start(self):
        self.process.cpu_affinity(self.cpus)



class ICL_Reg(RegistrationModule):
    def __init__(self, config):
        super().__init__(config)

    def load_pretrain(self, ckpt_path):
        state_dict = torch.load(ckpt_path)["state_dict"]
        fixed_state_dict = {
            key.replace("net.", ""): value for key, value in state_dict.items()
        }
        self.net.load_state_dict(fixed_state_dict)

    def forward(self, batch):
        fixed_image = batch["fixed_image"]
        fixed_label = batch["fixed_label"]

        moving_image = batch["moving_image"]
        moving_label = batch["moving_label"]

        # binarise the labels
        fixed_label = (fixed_label > 0).float()
        moving_label = (moving_label > 0).float()

        fixed_keypoints = batch["fixed_keypoints"]
        moving_keypoints = batch["moving_keypoints"]

        batch_size = fixed_image.shape[0]

        DDF_AB = self.net(torch.cat((fixed_image, moving_image), dim=1)).float()  # A->B
        DDF_BA = self.net(torch.cat((moving_image, fixed_image), dim=1)).float()  # B->A
        
        if fixed_keypoints is not None:
            with torch.no_grad():
                offset = torch.as_tensor(fixed_image.shape[-3:]).to(fixed_keypoints.device) / 2
                offset = offset[None][None]
                ddf_keypoints = torch.flip((fixed_keypoints - offset) / offset, (-1,))
            ddf_keypoints = (
                F.grid_sample(DDF_BA, ddf_keypoints.view(batch_size, -1, 1, 1, 3))
                .view(batch_size, 3, -1)
                .permute((0, 2, 1))
            )
        else:
            ddf_keypoints = None

        pred_image = self.warp_layer(moving_image, DDF_BA)
        pred_label = self.warp_layer(moving_label, DDF_BA)

        return DDF_AB, DDF_BA, ddf_keypoints, pred_image, pred_label
    
    def training_step(self, batch):
        DDF_AB, DDF_BA, ddf_keypoints, pred_image, pred_label = self(batch)

        fixed_label = (batch['fixed_label'] > 0).float()
        moving_label = (batch['moving_label'] > 0).float()

        loss = self.loss(DDF_AB, DDF_BA,
                        batch['fixed_image'], batch['moving_image'],
                        fixed_label, moving_label,)
        
        tre_before = self.tre(batch['fixed_keypoints'], batch['moving_keypoints'])
        tre_after = self.tre(batch['fixed_keypoints'] + ddf_keypoints, batch['moving_keypoints'])

        self.dice_metric_before(y_pred=fixed_label, y=moving_label)
        self.dice_metric_after(y_pred=pred_label, y=fixed_label)

        dice_before = self.dice_metric_before.aggregate().item()
        dice_after = self.dice_metric_after.aggregate().item()

        train_loss_dict = dict()
        train_loss_dict['train_tre_before'] = tre_before
        train_loss_dict['train_tre_after'] = tre_after
        train_loss_dict['train_loss'] = loss
        train_loss_dict['train_dice_before'] = dice_before
        train_loss_dict['train_dice_after'] = dice_after

        for loss_name, loss in train_loss_dict.items():
            self.log(
                loss_name,
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        self.dice_metric_before.reset()
        self.dice_metric_after.reset()  

        return {
            "loss": train_loss_dict['train_loss'],
        }

    def validation_step(self, batch, batch_idx):
        DDF_AB, DDF_BA, ddf_keypoints, pred_image, pred_label = self(batch)

        fixed_label = (batch['fixed_label'] > 0).float()
        moving_label = (batch['moving_label'] > 0).float()

        tre_before = self.tre(batch['fixed_keypoints'], batch['moving_keypoints'])
        tre_after = self.tre(batch['fixed_keypoints'] + ddf_keypoints, batch['moving_keypoints'])
        self.val_tre_before.append(tre_before)
        self.val_tre_after.append(tre_after)

        self.dice_metric_before(y_pred=fixed_label, y=moving_label)
        self.dice_metric_after(y_pred=pred_label, y=fixed_label)

    def on_validation_epoch_end(self):
        dice_before = self.dice_metric_before.aggregate().item()
        dice_after = self.dice_metric_after.aggregate().item()

        self.log(
            "val_dice",
            dice_after,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.dice_metric_before.reset()
        self.dice_metric_after.reset()

        tre_before = sum(self.val_tre_before) / len(self.val_tre_before)
        tre_after = sum(self.val_tre_after) / len(self.val_tre_after)

        self.log(
            "val_tre",
            tre_after,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.val_tre_before = []
        self.val_tre_after = []


class ICL_RegVessels(RegistrationModule):
    def __init__(self, config):
        super().__init__(config)


    def forward(self, batch_data):
            fixed_image = batch_data["fixed_image"]
            moving_image = batch_data["moving_image"]
            fixed_keypoints = batch_data["fixed_keypoints"]
            moving_keypoints = batch_data["moving_keypoints"]
            moving_label = batch_data["moving_label"]
            fixed_label = batch_data["fixed_label"]
            fixed_vessels = batch_data["fixed_vessels"]
            moving_vessels = batch_data["moving_vessels"]

            batch_size = fixed_image.shape[0]

            phi_AB = self.net(torch.cat((fixed_image, moving_image), dim=1)).float()
            phi_BA = self.net(torch.cat((moving_image, fixed_image), dim=1)).float()

            if fixed_keypoints is not None:
                with torch.no_grad():
                    offset = torch.as_tensor(fixed_image.shape[-3:]).to(fixed_keypoints.device) / 2
                    offset = offset[None][None]
                    ddf_keypoints = torch.flip((fixed_keypoints - offset) / offset, (-1,))
                ddf_keypoints = (
                    F.grid_sample(phi_AB, ddf_keypoints.view(batch_size, -1, 1, 1, 3)) 
                    .view(batch_size, 3, -1)
                    .permute((0, 2, 1))
                )
            else:
                ddf_keypoints = None

            return phi_AB, phi_BA, ddf_keypoints


    def training_step(self, batch):
        phi_AB, phi_BA, ddf_keypoints = self(batch)

        tre_before = self.tre(batch['fixed_keypoints'], batch['moving_keypoints'])
        tre_after = self.tre(batch['fixed_keypoints'], batch['moving_keypoints'] + ddf_keypoints)

        print(type(batch['fixed_vessels']), type(batch['moving_vessels']))
        print(type(batch['fixed_label']), type(batch['moving_label']))

         # make vessels binary, 0 values remain 0, all other values become 1
        vessels_fixed = (batch['fixed_vessels'] > 0).float()
        vessels_moving = (batch['moving_vessels'] > 0).float()

        print(type(vessels_fixed), type(vessels_moving))

        loss = self.loss(phi_AB, phi_BA, 
                         batch['fixed_image'], batch['moving_image'], 
                         batch['fixed_label'], batch['moving_label'],
                         vessels_fixed, vessels_moving)

        train_loss_dict = dict()
        train_loss_dict['train_tre_before'] = tre_before
        train_loss_dict['train_tre_after'] = tre_after
        train_loss_dict['train_loss'] = loss

        for loss_name, loss in train_loss_dict.items():
            self.log(
                loss_name,
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return {
            "loss": train_loss_dict['train_loss'],
        }
    
    def validation_step(self, batch, batch_idx):
        phi_AB, phi_BA, ddf_keypoints = self(batch)

        tre_before = self.tre(batch['fixed_keypoints'], batch['moving_keypoints'])
        tre_after = self.tre(batch['fixed_keypoints'], batch['moving_keypoints'] + ddf_keypoints)
        self.val_tre_before.append(tre_before)
        self.val_tre_after.append(tre_after)

        pred_mask = self.warp_layer(batch['moving_vessels'], phi_BA)
        self.dice_metric_before(y_pred=batch['moving_vessels'], y=batch['fixed_vessels'])
        self.dice_metric_after(y_pred=pred_mask, y=batch['fixed_vessels'])

    def on_validation_epoch_end(self):
        dice_before = self.dice_metric_before.aggregate().item()
        dice_after = self.dice_metric_after.aggregate().item()

        self.log(
            "val_dice",
            dice_after,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.dice_metric_before.reset()
        self.dice_metric_after.reset()

        tre_before = sum(self.val_tre_before) / len(self.val_tre_before)
        tre_after = sum(self.val_tre_after) / len(self.val_tre_after)

        self.log(
            "val_tre",
            tre_after,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.val_tre_before = []
        self.val_tre_after = []
