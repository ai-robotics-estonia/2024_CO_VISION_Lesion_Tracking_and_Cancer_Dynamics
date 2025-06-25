import torch
import numpy as np


class DeepSupervisionBinaryFocalLoss(torch.nn.Module):
    def __init__(self, focal_loss_fn):
        super().__init__()

        self.focal_loss_fn = focal_loss_fn

    def forward(
        self, inputs: list[torch.Tensor], targets: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute focal loss for binary classification problem (deep supervision outputs)
        Args:
            inputs: list of tensors [[B,1,D,H,W], [B,1,D/2,H/2,W/2], ...]
            targets: list of tensors [[B,1,D,H,W], [B,1,D/2,H/2,W/2], ...]

        Returns:

        """
        input_shapes = np.array([pred.shape for pred in inputs])
        target_shapes = np.array([gt.shape for gt in targets])
        assert np.all(
            input_shapes == target_shapes
        ), f"Inputs and Targets have different shapes: {input_shapes.tolist()}\nand\n{target_shapes.tolist()}"

        losses = [
            self.focal_loss_fn(pred, gt) * (2 ** (-i))
            for i, (pred, gt) in enumerate(zip(inputs, targets))
        ]

        return sum(losses)
