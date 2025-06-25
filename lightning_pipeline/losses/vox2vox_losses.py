import torch
import torch.nn.functional as F
from monai.losses import DiceLoss

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

class LNCC(torch.nn.Module):
    def __init__(self, sigma = 1):
        super().__init__()
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)

    def forward(self, 
                image_A, 
                image_B,
                mask_A,
                ):
        I = image_A
        J = image_B
        assert I.shape == J.shape, "The shape of image I and J sould be the same."

        lncc = \
        ( 
            1
            - (self.blur(I * J) - (self.blur(I) * self.blur(J)))
            / torch.sqrt(
                (torch.relu(self.blur(I * I) - self.blur(I) ** 2) + 0.00001)
                * (torch.relu(self.blur(J * J) - self.blur(J) ** 2) + 0.00001)
            )
        )

        mask_area = mask_A
        lncc_loss = torch.sum(lncc * mask_area) / (torch.sum(mask_area) + 1e-8)

        return lncc_loss

class GradSmoothLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ddf: torch.Tensor) -> ...:
        dx = ddf[:, :, 1:, :, :] - ddf[:, :, :-1, :, :]
        dy = ddf[:, :, :, 1:, :] - ddf[:, :, :, :-1, :]
        dz = ddf[:, :, :, :, 1:] - ddf[:, :, :, :, :-1]

        loss = torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)

        return loss
    
class VesselLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()

    def erode_vessels(self, mask_A, mask_B):
        pass

    def forward(self, mask_A, mask_B):
        dice_loss = self.dice_loss(mask_A, mask_B)
        return dice_loss

class LungLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()

    def forward(self, mask_A, mask_B):
        dice_loss = self.dice_loss(mask_A, mask_B)
        return dice_loss
