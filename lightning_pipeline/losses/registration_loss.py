# Implementation is based on MONAI learn2reg notebook:
# https://github.com/Project-MONAI/tutorials/blob/main/3d_registration/learn2reg_nlst_paired_lung_ct.ipynb

import torch
import numpy as np

from torch.nn import MSELoss
from monai.losses import BendingEnergyLoss, DiceLoss
from monai.networks.blocks import Warp
from monai.transforms.utils import create_grid

import torch.nn.functional as F

class TRE(torch.nn.Module):
    """
    Computes target registration error (TRE) loss for keypoint matching.

    Args:
        vx: voxel size of the input images, 3D numpy array for (z, y, x) axes.
    """

    def __init__(self, vx: torch.Tensor = None):
        super().__init__()

        self.vx = vx

    def forward(self, fixed, moving):
        if self.vx is None:
            return ((fixed - moving) ** 2).sum(-1).sqrt().nanmean()
        else:
            return ((fixed - moving).mul(self.vx) ** 2).sum(-1).sqrt().nanmean()


class RegMultiTargetLoss(torch.nn.Module):
    """
    Multi-target loss from MONAI tutorial:
        - TRE as main loss component
        - Parametrizable weights for further (optional) components: MSE/BendingEnergy/Dice loss
        Note: Might require "calibration" of lambda weights for the multi-target components,
        e.g. by making a first trial run, and manually setting weights to account for different magnitudes
    """

    def __init__(
        self,
        lam_t: float = 1.0,
        lam_m: float = 0.0,
        lam_r: float = 0.0,
        lam_l: float = 0.0,
        vx: list = None,
    ):
        super().__init__()
        self.lam_t = lam_t
        self.lam_m = lam_m
        self.lam_r = lam_r
        self.lam_l = lam_l
        self.vx = vx if vx is not None else None

        if lam_t > 0:
            self.tre = TRE(vx=self.vx)
        if lam_m > 0:
            self.mse = MSELoss()
        if lam_r > 0:
            self.regularization = BendingEnergyLoss()
        if lam_l > 0:
            self.label_loss = DiceLoss()

    def forward(
        self,
        fixed_image: torch.Tensor,
        pred_image: torch.Tensor,
        fixed_label: torch.Tensor,
        pred_label: torch.Tensor,
        ddf_image: torch.Tensor,
        fixed_keypoints: torch.Tensor = None,
        pred_keypoints: torch.Tensor = None,
    ):
        t = (
            self.tre(fixed_keypoints, pred_keypoints)
            if (self.lam_t > 0 and fixed_keypoints is not None)
            else 0.0
        )
        l = self.label_loss(pred_label, fixed_label) if self.lam_l > 0 else 0.0
        m = self.mse(fixed_image, pred_image) if self.lam_m > 0 else 0.0
        r = self.regularization(ddf_image) if self.lam_r > 0 else 0.0
        return self.lam_t * t + self.lam_m * m + self.lam_r * r + self.lam_l * l


def _get_gaussian_kernel1d(kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    return kernel1d

def gaussian_blur(tensor, kernel_size, sigma, padding="same"):
    kernel1d = _get_gaussian_kernel1d(kernel_size=kernel_size, sigma=sigma).to(
        tensor.device, dtype=tensor.dtype
    )
    out = tensor
    group = tensor.shape[1]

    if len(tensor.shape) - 2 == 1:
        out = torch.conv1d(out, kernel1d[None, None, :].expand(group,-1,-1), padding="same", groups=group)
    elif len(tensor.shape) - 2 == 2:
        out = torch.conv2d(out, kernel1d[None, None, :, None].expand(group,-1,-1,-1), padding="same", groups=group)
        out = torch.conv2d(out, kernel1d[None, None, None, :].expand(group,-1,-1,-1), padding="same", groups=group)
    elif len(tensor.shape) - 2 == 3:
        out = torch.conv3d(out, kernel1d[None, None, :, None, None].expand(group,-1,-1,-1,-1), padding="same", groups=group)
        out = torch.conv3d(out, kernel1d[None, None, None, :, None].expand(group,-1,-1,-1,-1), padding="same", groups=group)
        out = torch.conv3d(out, kernel1d[None, None, None, None, :].expand(group,-1,-1,-1,-1), padding="same", groups=group)

    return out

class LNCC():
    def __init__(self, sigma = 1):
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)

    def __call__(self, image_A, image_B):
        I = image_A
        J = image_B
        assert I.shape == J.shape, "The shape of image I and J sould be the same."

        return torch.mean(
            1
            - (self.blur(I * J) - (self.blur(I) * self.blur(J)))
            / torch.sqrt(
                (torch.relu(self.blur(I * I) - self.blur(I) ** 2) + 0.00001)
                * (torch.relu(self.blur(J * J) - self.blur(J) ** 2) + 0.00001)
            )
        )


class ICONLoss(torch.nn.Module):
    def __init__(self, 
                 lambda_gradicon=128):
        super().__init__()
        self.lncc_loss = LNCC(sigma=1)
        self.lambda_gradicon = lambda_gradicon
        self.warp_layer = Warp()
        self.label_loss = DiceLoss()

    def compute_IM(self, size, batch_size=1):
        coords = torch.meshgrid([torch.linspace(-1, 1, s, device='cuda') for s in size], indexing='ij')
        grid = torch.stack(coords, dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        return grid
    
    def forward(self,
                phi_AB: torch.Tensor, 
                phi_BA: torch.Tensor,
                image_A: torch.Tensor,
                image_B: torch.Tensor,
                label_A: torch.Tensor,
                label_B: torch.Tensor) -> torch.Tensor:

        identity_grid = self.compute_IM(phi_AB.shape[2:], batch_size=phi_AB.shape[0])

        Iepsilon = (
            identity_grid
            + torch.randn(*identity_grid.shape).to(image_A.device)
            * 1
            / identity_grid.shape[-1]
        )
        
        forward_warped = self.warp_layer(self.warp_layer(Iepsilon, phi_AB), phi_BA)
        backward_warped = self.warp_layer(self.warp_layer(Iepsilon, phi_BA), phi_AB)

        ic_loss = F.mse_loss(Iepsilon, forward_warped) + F.mse_loss(Iepsilon, backward_warped)

        warped_A = self.warp_layer(image_A, phi_AB)
        warped_B = self.warp_layer(image_B, phi_BA)

        sim_loss = self.lncc_loss(warped_A, image_B) + self.lncc_loss(image_A, warped_B)

        warped_label_A = self.warp_layer(label_A, phi_AB)
        warped_label_B = self.warp_layer(label_B, phi_BA)

        dice_loss = self.label_loss(warped_label_A, label_B) + self.label_loss(label_A, warped_label_B)

        return self.lambda_gradicon * ic_loss + sim_loss + dice_loss

class GradICONLoss(torch.nn.Module):
    def __init__(self, 
                 lambda_gradicon=0.6,
                 lambda_dice=0.3,
                 lambda_lncc=0.1):
        super().__init__()
        self.lambda_gradicon = lambda_gradicon
        self.lambda_dice = lambda_dice
        self.lambda_lncc = lambda_lncc
        self.warp_layer = Warp()
        self.delta = 0.001
        self.label_loss = DiceLoss()
        self.lncc_loss = LNCC(sigma=1)

    def compute_IM(self, size, batch_size=1):
        coords = torch.meshgrid([torch.linspace(-1, 1, s, device='cuda') for s in size], indexing='ij')
        grid = torch.stack(coords, dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        return grid
    
    def forward(self,
                phi_AB: torch.Tensor, 
                phi_BA: torch.Tensor,
                image_A: torch.Tensor,
                image_B: torch.Tensor,
                label_A: torch.Tensor,
                label_B: torch.Tensor) -> torch.Tensor:

        identity_grid = self.compute_IM(phi_AB.shape[2:], batch_size=phi_AB.shape[0])

        # isIdentity = True
        Iepsilon = (
            identity_grid
            + torch.randn(*identity_grid.shape).to(image_A.device)
            * 1
            / identity_grid.shape[-1]
        )
        
        # If isIdentity = True, according to soruce code:
            # def transform(coordinates):
            #     if hasattr(coordinates, "isIdentity") and coordinates.shape == tensor_of_displacements.shape:
            #         return coordinates + tensor_of_displacements
            #     return coordinates + displacement_field(coordinates)

        # we should simply add the displacement field to the coordinates

        # this is wrong actually:
            # apprx_Iepsilon = self.warp_layer(self.warp_layer(Iepsilon, phi_AB), phi_BA)
        
        apprx_Iepsilon = (Iepsilon + phi_BA) + phi_AB
        ic_error = Iepsilon - apprx_Iepsilon

        dx = torch.Tensor([[[[[self.delta]]], [[[0.0]]], [[[0.0]]]]]).to(
            identity_grid.device
        )
        dy = torch.Tensor([[[[[0.0]]], [[[self.delta]]], [[[0.0]]]]]).to(
            identity_grid.device
        )
        dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[self.delta]]]]).to(
            identity_grid.device
        )
        direction_vectors = (dx, dy, dz)

        direction_losses = []
        for d in direction_vectors:
            apprx_Iepsilon_d = (((Iepsilon + self.delta) + phi_BA) + phi_AB)
            #this is also wrong:
                # apprx_Iepsilon_d = self.warp_layer(self.warp_layer(Iepsilon + d, phi_AB), phi_BA)
            inverse_consistency_error_d = Iepsilon + d - apprx_Iepsilon_d 
            grad_d_icon_error = (
                ic_error - inverse_consistency_error_d
            ) / self.delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        warped_A = self.warp_layer(image_A, phi_AB)
        warped_B = self.warp_layer(image_B, phi_BA)

        sim_loss = self.lncc_loss(warped_A, image_B) + self.lncc_loss(image_A, warped_B)

        warped_label_A = self.warp_layer(label_A, phi_AB)
        warped_label_B = self.warp_layer(label_B, phi_BA)

        dice_loss = self.label_loss(warped_label_A, label_B) + self.label_loss(label_A, warped_label_B)

        return self.lambda_gradicon * inverse_consistency_loss + self.lambda_lncc * sim_loss + self.lambda_dice * dice_loss


