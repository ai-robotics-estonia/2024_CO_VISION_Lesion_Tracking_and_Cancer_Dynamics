import torch
import numpy as np
import torch.nn.functional as F


class BoundaryWeightedBCELoss(torch.nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def boundary_weighted_bce_loss(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
        weight = 1 + 5 * torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_bce = (weight * bce).flatten()

        if self.reduction == 'mean':
            weighted_bce = weighted_bce.mean()
        
        return weighted_bce

    def forward(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
        input_shapes = np.array([pred.shape for pred in inputs])
        target_shapes = np.array([gt.shape for gt in targets])

        assert np.all(input_shapes == target_shapes), \
            f'Inputs and Targets have different shapes: {input_shapes.tolist()}\nand\n{target_shapes.tolist()}'

        return torch.sum(self.boundary_weighted_bce_loss(inputs, targets))
