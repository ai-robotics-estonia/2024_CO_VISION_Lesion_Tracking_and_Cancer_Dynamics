"""Model training pipeline

Usage: python test.py --config-name=train_monai_detector.yaml +ckpt_path=<path-to-ckpt>
"""


import torch
import hydra
from omegaconf import OmegaConf
from pathlib import Path
from visualise_disp import plot_disp # Replace with your actual visualization function
from einops import rearrange
from lightning_modules.models.registration_module import ICL_Reg
from corrfield.corrfield import corrfield
from monai.networks.blocks import Warp


def generate_cutom_ddf(fix_img: torch.Tensor, fix_mask: torch.Tensor, mov_img: torch.Tensor, mov_mask: torch.Tensor ) -> torch.Tensor:
    ddf, _, _ = corrfield(fix_img, fix_mask, mov_img, 2.5, 150, 5, 1, 3, 0, 1, 1.4, 0.8, True, [16, 8], [6, 3], [2, 1], [3, 2], ['n', 'n'])
    ddf_masked = ddf.detach().cpu().clone()
    print(ddf_masked.shape)
    print(mov_mask.squeeze(1).shape)
    ddf_masked[~mov_mask.squeeze(1).to(torch.bool)] = 0

    return ddf_masked

def adjust_ddf(ddf_masked: torch.Tensor, mov_mask: torch.Tensor) -> torch.Tensor:
    batch_size = mov_mask.shape[0]
    ddf_masked_adjusted = ddf_masked.clone().repeat(batch_size, 1, 1, 1, 1)
    ddf_masked_adjusted[~mov_mask.squeeze(1).to(torch.bool)] = 0
    return ddf_masked_adjusted


# Load config and model
cfg = OmegaConf.load("/gpfs/space/projects/BetterMedicine/illia/bm-ai-pipelines/common/ocs/lightning_pipeline/lightning_logs/registration_NLST_ICON/version_92/hparams.yaml")  # Update path if necessary
model = ICL_Reg(cfg)
vessels_ckpt = '/gpfs/space/projects/BetterMedicine/illia/bm-ai-pipelines/common/ocs/lightning_pipeline/lightning_logs/registration_NLST_ICON/version_94/checkpoints/epoch=464-val_tre=2.123-val_dice=0.801.ckpt'
mask_ckpt = '/gpfs/space/projects/BetterMedicine/illia/bm-ai-pipelines/common/ocs/lightning_pipeline/lightning_logs/registration_NLST_ICON/version_92/checkpoints/epoch=104-val_tre=3.237-val_dice=0.846.ckpt'
model.load_pretrain(ckpt_path=vessels_ckpt)
model.cuda()
model.eval()

warper = Warp()

# Load a single validation sample
datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.setup(stage="fit")
val_loader = datamodule.val_dataloader()

# Get one batch
data_iter = iter(val_loader)
sample_batch = next(data_iter)

ddf_masked = generate_cutom_ddf(sample_batch['fixed_image'], sample_batch['fixed_label'], sample_batch['moving_image'], sample_batch['moving_label'])
print(ddf_masked.shape)

mses = []
mses_corr = []
for i, batch in enumerate(val_loader):
    custom_ddf = None
    for key in batch:
        ddf_masked_adjusted = adjust_ddf(ddf_masked, batch['moving_label'])
        ddf_masked_adjusted = rearrange(ddf_masked_adjusted, 'B H W D C -> B C H W D')
        custom_ddf = ddf_masked_adjusted

        batch['moving_label'] = (warper(batch['moving_label'].cuda(), ddf_masked_adjusted.cuda()) > 0).float()
        batch['moving_image'] = warper(batch['moving_image'].cuda(), ddf_masked_adjusted.cuda())

        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()

    ddf_corrfield, _, _ = corrfield(batch['fixed_image'], batch['fixed_label'], batch['moving_image'], 2.5, 150, 5, 1, 3, 0, 1, 1.4, 0.8, True, [16, 8], [6, 3], [2, 1], [3, 2], ['n', 'n'])
    ddf_corrfield = ddf_corrfield.detach().cpu()
    ddf_corrfield[~batch['moving_label'].squeeze(1).to(torch.bool)] = 0
    ddf_corrfield = rearrange(ddf_corrfield, 'B H W D C -> B C H W D')

    ddf_image, _, _, _, _ = model(batch)
    ddf_image = ddf_image.detach().cpu()
    ddf_image[~batch['moving_label'].to(torch.bool).expand_as(ddf_image)] = 0

    ddf_mse = torch.mean((custom_ddf - ddf_image).pow(2))
    ddf_mse_corrfield = torch.mean((custom_ddf - ddf_corrfield).pow(2))

    print(f"For batch #{i} of size {batch['fixed_image'].shape[0]}, MSE between DL_ddf and custom_defined_ddf is {ddf_mse}")
    print(f"For batch #{i} of size {batch['fixed_image'].shape[0]}, MSE between CorrField_ddf and custom_defined_ddf is {ddf_mse_corrfield}")
    mses.append(ddf_mse)
    mses_corr.append(ddf_mse_corrfield)

print(f"Average MSE for DL is {sum(mses) / len(mses)}")
print(f"Average MSE for CorrField is {sum(mses_corr) / len(mses_corr)}")

# print(batch.keys())

# # Move batch to GPU
# for key in batch:
#     if isinstance(batch[key], torch.Tensor):
#         batch[key] = batch[key].cuda()

# # Run model inference
# ddf_image, _, _, _, _ = model(batch)  # Only extracting DDF

# # Convert DDF to CPU for visualization
# ddf_image = ddf_image.detach().cpu().numpy()

# ddf_image = rearrange(ddf_image, 'B C H W D -> B H W D C')
# ddf_image = ddf_image[0]
# mask = batch['fixed_label'][0][0]

# print(batch['fixed_image'].shape)
# print(batch['fixed_label'].shape)
# print(batch['moving_image'].shape)

# disp, _, _ = corrfield(batch['fixed_image'][0].unsqueeze(0), batch['fixed_label'][0].unsqueeze(0), batch['moving_image'][0].unsqueeze(0),2.5, 150, 5, 1, 3, 0, 1, 1.4, 0.8, True, [16,8], [6,3], [2,1], [3,2], ['n','n'])
# print(disp.shape)
# # Visualize DDF
# plot_disp(ddf_image, mask, 'disp_vessels.html')  # Update path if necessary
# plot_disp(disp[0].cpu().numpy(), mask, 'disp_mask_corrfield.html')  # Update path if necessary