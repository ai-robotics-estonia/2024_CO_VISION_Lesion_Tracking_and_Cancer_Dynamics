from models.vox2vox import Discriminator, GeneratorUNetGlobal
import torch
import hydra
from omegaconf import OmegaConf
from pathlib import Path
from einops import rearrange
from lightning_modules.models.registration_module import ICL_Reg
from corrfield.corrfield import corrfield
from monai.networks.blocks import Warp
import numpy as np
from typing import Tuple, Any, Dict, List, Union
from tqdm import tqdm

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import pyplot as plt

def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def generate_cutom_ddf(fix_img: torch.Tensor, fix_mask: torch.Tensor, mov_img: torch.Tensor, mov_mask: torch.Tensor ) -> torch.Tensor:
    ddf, _, _ = corrfield(fix_img, fix_mask, mov_img, 2.5, 150, 5, 1, 3, 0, 1, 1.4, 0.8, True, [16, 8], [6, 3], [2, 1], [3, 2], ['n', 'n'])
    ddf_masked = ddf.detach().cpu().clone()
    ddf_masked[~mov_mask.squeeze(1).to(torch.bool)] = 0

    return ddf_masked

def adjust_ddf(ddf_masked: torch.Tensor, mov_mask: torch.Tensor) -> torch.Tensor:
    batch_size = mov_mask.shape[0]
    ddf_masked_adjusted = ddf_masked.clone().repeat(batch_size, 1, 1, 1, 1)
    ddf_masked_adjusted[~mov_mask.squeeze(1).to(torch.bool)] = 0
    return ddf_masked_adjusted

def compute_r2(y_pred: torch.Tensor, y_true: torch.Tensor, y_true_mean: torch.Tensor = 0) -> float:
    r2 = 1 - (torch.sum((y_true - y_pred).pow(2)) / torch.sum((y_true - y_true_mean).pow(2)))
    return r2.item()


def sample_all_test_voxel_volumes(path,
                                  batch,
                                  ddf, # (B, C, H, W, D)
                                  device = 'cuda'):
    """Saves a generated sample from the validation set"""
    print("\n\nSampling images...")

    real_A_glob = batch["fixed_image"].cpu()
    real_B_glob = batch["moving_image"].cpu()
    fixed_keypoints = batch["fixed_keypoints"].cpu().numpy()[0]
    moving_keypoints = batch["moving_keypoints"].cpu().numpy()[0]
    fixed_label = batch["fixed_label"].cpu().numpy()[0][0]
    
    valid_indices = []
    for idx, point in enumerate(fixed_keypoints):
        at_lungs = fixed_label[int(point[0]), int(point[1]), int(point[2])]
        if at_lungs:
            buffer = 5
            x, y, z = int(point[0]), int(point[1]), int(point[2])
            x_min, x_max = max(0, x-buffer), min(fixed_label.shape[0], x+buffer)
            y_min, y_max = max(0, y-buffer), min(fixed_label.shape[1], y+buffer)
            z_min, z_max = max(0, z-buffer), min(fixed_label.shape[2], z+buffer)
            
            region = fixed_label[x_min:x_max, y_min:y_max, z_min:z_max]
            if np.all(region > 0):
                valid_indices.append(idx)

    np.random.seed(1337)

    idcs = np.random.randint(0, len(valid_indices), size=3)
    fp = fixed_keypoints[valid_indices][idcs]
    mp = moving_keypoints[valid_indices][idcs]

    fake_DDF_glob = ddf.cpu()
    fake_A_glob = warper(real_B_glob, fake_DDF_glob) # DDF(MV) -> FX

    fake_DDF_glob = fake_DDF_glob.cpu().numpy()
    fake_DDF_glob = rearrange(fake_DDF_glob, 'B C H W D -> B H W D C')[0]
    
    fig, axs = plt.subplots(len(fp), 3, figsize=(12, 4*len(fp)))

    q_for_colorbar = None
    for k in range(len(fp)):
        U, V = fake_DDF_glob[:, :, int(fp[k][2]), 0], fake_DDF_glob[:, :, int(fp[k][2]), 1]
        Y, X = np.meshgrid(np.arange(U.shape[0]), np.arange(U.shape[1]), indexing='ij')
        magnitude = np.sqrt(U**2 + V**2)

        real_B = real_B_glob[0, 0, :, :, int(mp[k][2])].cpu().numpy()
        real_A = real_A_glob[0, 0, :, :, int(fp[k][2])].cpu().numpy()
        fake_A = fake_A_glob[0, 0, :, :, int(fp[k][2])].cpu().numpy()

        axs[k, 0].imshow(real_B, cmap='gray')
        axs[k, 1].imshow(real_A, cmap='gray')
        axs[k, 2].imshow(fake_A, cmap='gray')

        rect_size = 16
        half_size = rect_size // 2

        x, y = int(mp[k][1]), int(mp[k][0])
        rect = plt.Rectangle((x - half_size, y - half_size), rect_size, rect_size, 
                            fill=False, color='red', linewidth=1)
        axs[k, 0].add_patch(rect)

        x, y = int(fp[k][1]), int(fp[k][0])
        rect = plt.Rectangle((x - half_size, y - half_size), rect_size, rect_size, 
                            fill=False, color='red', linewidth=1)
        axs[k, 1].add_patch(rect)

        x, y = int(fp[k][1]), int(fp[k][0])
        rect = plt.Rectangle((x - half_size, y - half_size), rect_size, rect_size, 
                            fill=False, color='red', linewidth=1)
        axs[k, 2].add_patch(rect)
        
        axs[k, 0].scatter(mp[k][1], mp[k][0], color='red', s=10, marker='o')
        axs[k, 1].scatter(fp[k][1], fp[k][0], color='red', s=10, marker='o')
        axs[k, 2].scatter(fp[k][1], fp[k][0], color='red', s=10, marker='o')

        step = 3
        q = axs[k, 0].quiver(X[::step, ::step], Y[::step, ::step], 
                        U[::step, ::step], V[::step, ::step],
                        magnitude[::step, ::step], alpha=0.9)
        
        axs[k, 0].set_title(f'MOV@[{int(mp[k][0])}, {int(mp[k][1])}, {int(mp[k][2])}]', fontsize=9)
        axs[k, 1].set_title(f'FIX@[{int(fp[k][0])}, {int(fp[k][1])}, {int(fp[k][2])}]', fontsize=9)
        axs[k, 2].set_title(f'FAKE_FIX@[{int(fp[k][0])}, {int(fp[k][1])}, {int(fp[k][2])}]', fontsize=9)

        for ax in axs[k]:
            ax.axis('off')

        if k == 0:
            q_for_colorbar = q
    
    cbar_ax = fig.add_axes([0.02, 0.25, 0.02, 0.5])
    if q_for_colorbar is not None:
        cbar = fig.colorbar(q_for_colorbar, cax=cbar_ax)
        cbar.set_label('')
        
    fig.subplots_adjust(wspace=0.05, hspace=0.01, left=0.08, top=0.95, bottom=0.05)
    plt.savefig(path, 
                dpi=500, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

def vis_ddf(ddf: torch.Tensor, image: torch.Tensor, name: str, slice_idx: int = 128, sample_factor: int = 10, alpha: float = 0.5) -> None:
    plt.close()
    ct_slice = image[0][0][:, :, slice_idx].detach().cpu().numpy()
    ddf_slice = ddf[0][:, :, slice_idx].detach().cpu().numpy()

    X, Y = np.meshgrid(np.arange(0, ct_slice.shape[1]), np.arange(0, ct_slice.shape[0]))
    U, V = ddf_slice[:, :, 0], ddf_slice[:, :, 1]

    X_sub = X[::sample_factor, ::sample_factor]
    Y_sub = Y[::sample_factor, ::sample_factor]

    U_sub = U[::sample_factor, ::sample_factor]
    V_sub = V[::sample_factor, ::sample_factor]

    magnitude = np.sqrt(U**2 + V**2)
    magnitude_sub = np.sqrt(U_sub**2 + V_sub**2)

    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=np.percentile(magnitude, 95))
    cmap.set_bad(color='black')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(ct_slice, cmap='gray')

    ax.quiver(X_sub, Y_sub, U_sub, V_sub, magnitude_sub, cmap=cmap, scale=1, scale_units='xy', angles='xy', alpha=alpha)
    
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(magnitude)
    fig.colorbar(sm, ax=ax)

    ax.set_title(f'CT slice at {slice_idx} idx')
    plt.savefig(f'/gpfs/space/projects/BetterMedicine/illia/bm-ai-pipelines/common/ocs/lightning_pipeline/sample_eval_images/vis_ddf_{slice_idx}_{name}.png')

cfg = OmegaConf.load("/gpfs/space/projects/BetterMedicine/illia/bm-ai-pipelines/common/ocs/lightning_pipeline/lightning_logs/registration_NLST_ICON/version_54/hparams.yaml")  # Update path if necessary
cfg.datamodule.config.dataloaders.val_dataloader.batch_size = 1
cfg.datamodule.config.dataloaders.train_dataloader.batch_size = 1

for transform in cfg.datamodule.config.transforms.train_transforms.transforms:
    if "_target_" in transform and "Resized" in transform["_target_"]:
        transform.spatial_size = [128, 128, 128]

for transform in cfg.datamodule.config.transforms.val_transforms.transforms:
    if "_target_" in transform and "Resized" in transform["_target_"]:
        transform.spatial_size = [128, 128, 128]

dl_vessel_weights = '/gpfs/space/projects/BetterMedicine/illia/bm-ai-pipelines/common/ocs/lightning_pipeline/lightning_logs/registration_NLST_ICON/version_54/checkpoints/epoch=374-val_tre=3.416-val_dice=0.172.ckpt'
gen_weights = '/gpfs/space/projects/BetterMedicine/illia/vox2vox_registration/saved_models/[LNCC_10.0, GradSmooth_0.0, Vessel_10.0, Lung_0.0, ADV_1.0, Sigma_1.0]_[NLST]_[augs]_[1input_D]/generator_global_70.pth'
disc_weights = '/gpfs/space/projects/BetterMedicine/illia/vox2vox_registration/saved_models/[LNCC_10.0, GradSmooth_0.0, Vessel_10.0, Lung_0.0, ADV_1.0, Sigma_1.0]_[NLST]_[augs]_[1input_D]/discriminator_global_200.pth'

gen_ckpt = torch.load(gen_weights, weights_only=True)
disc_ckpt = torch.load(disc_weights, weights_only=True)

dl_model = ICL_Reg(cfg)
vox2vox_gen = GeneratorUNetGlobal()
warper = Warp()

dl_model.load_pretrain(ckpt_path=dl_vessel_weights)
dl_model.eval()
dl_model.cuda()

vox2vox_gen.load_state_dict(strip_module_prefix(gen_ckpt))
vox2vox_gen.eval()
vox2vox_gen.cuda()

datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.setup(stage="fit")

val_loader = datamodule.val_dataloader()
train_loader = datamodule.train_dataloader()

generated_ddfs = []
print('Generating custom DDFs')
for i, train_batch in tqdm(enumerate(train_loader)):
    if len(generated_ddfs) > 2:
        break
    fixed, moving = train_batch["fixed_image"].to('cuda'), train_batch["moving_image"].to('cuda')
    lung_A, lung_B = train_batch["fixed_label"].to('cuda'), train_batch["moving_label"].to('cuda')

    gen_ddf = generate_cutom_ddf(fixed, lung_A, moving, lung_B)
    generated_ddfs.append(gen_ddf)

print('Finished sampling images')
print('Starting evaluation')
corr_mse_total, corr_r2_total = [], []
dl_mse_total, dl_r2_total = [], []
vox2vox_mse_total, vox2vox_r2_total = [], []
for i, val_batch in tqdm(enumerate(val_loader)):
    print(f'Batch #{i}')
    for k, gen_ddf in enumerate(generated_ddfs):
        # print(f'GEN_DDF #{k}')
        # custom_adjusted = None
        for key in val_batch:
            # adjusted_custom_ddf = adjust_ddf(gen_ddf, val_batch['moving_label'])
            # adjusted_custom_ddf = rearrange(adjusted_custom_ddf, 'B H W D C -> B C H W D')
            # val_batch['fixed_label'] = (warper(val_batch['moving_label'].cuda(), adjusted_custom_ddf.cuda()) > 0).float()
            # val_batch['fixed_image'] = warper(val_batch['moving_image'].cuda(), adjusted_custom_ddf.cuda())
            # custom_adjusted = adjusted_custom_ddf

            if isinstance(val_batch[key], torch.Tensor):
                val_batch[key] = val_batch[key].cuda()
        
        ddf_corrfield, _, _ = corrfield(val_batch['fixed_image'], val_batch['fixed_label'], val_batch['moving_image'], 2.5, 150, 5, 1, 3, 0, 1, 1.4, 0.8, True, [16, 8], [6, 3], [2, 1], [3, 2], ['n', 'n'])
        ddf_corrfield = ddf_corrfield.detach().cpu()
        ddf_corrfield[~val_batch['moving_label'].squeeze(1).to(torch.bool)] = 0
        ddf_corrfield = rearrange(ddf_corrfield, 'B H W D C -> B C H W D')

        _, ddf_dl, _, _, _ = dl_model(val_batch)
        ddf_dl = ddf_dl.detach().cpu()
        ddf_dl[~val_batch['moving_label'].to(torch.bool).expand_as(ddf_dl)] = 0

        ddf_vox2vox, _ = vox2vox_gen(torch.cat([val_batch['moving_image'], val_batch['fixed_image']], dim=1))
        ddf_vox2vox = ddf_vox2vox.detach().cpu()
        ddf_vox2vox[~val_batch['moving_label'].to(torch.bool).expand_as(ddf_vox2vox)] = 0

        # mse_dl = torch.mean((custom_adjusted - ddf_dl).pow(2)).item()
        # mse_corrfield = torch.mean((custom_adjusted - ddf_corrfield).pow(2)).item()
        # mse_vox2vox = torch.mean((custom_adjusted - ddf_vox2vox).pow(2)).item()

        # r2_dl = compute_r2(ddf_dl, custom_adjusted)
        # r2_corrfield = compute_r2(ddf_corrfield, custom_adjusted)
        # r2_vox2vox = compute_r2(ddf_vox2vox, custom_adjusted)

        sample_all_test_voxel_volumes('./vox2vox_vis.png', 
                                      val_batch,
                                      ddf_vox2vox)
        
        sample_all_test_voxel_volumes('./corrfield_vis.png',
                                      val_batch,
                                      ddf_corrfield)
        
        sample_all_test_voxel_volumes('./dl_vis.png',
                                        val_batch,
                                        ddf_dl)
        
        break
    break

        # corr_mse_total.append(mse_corrfield)
        # corr_r2_total.append(r2_corrfield)

        # dl_mse_total.append(mse_dl)
        # dl_r2_total.append(r2_dl)

        # vox2vox_mse_total.append(mse_vox2vox)
        # vox2vox_r2_total.append(r2_vox2vox)


        # batch_res_str = f"Batch #{i} | GEN_DDF #{k} | MSE for DL: {mse_dl} | MSE for CorrField: {mse_corrfield} | MSE for Vox2Vox: {mse_vox2vox} | R^2 for DL: {r2_dl} | R^2 for CorrField: {r2_corrfield} | R^2 for Vox2Vox: {r2_vox2vox}"

        # print(batch_res_str)

# avg_corr_mse = sum(corr_mse_total) / len(corr_mse_total)
# avg_corr_r2 = sum(corr_r2_total) / len(corr_r2_total)

# avg_dl_mse = sum(dl_mse_total) / len(dl_mse_total)
# avg_dl_r2 = sum(dl_r2_total) / len(dl_r2_total)

# avg_vox2vox_mse = sum(vox2vox_mse_total) / len(vox2vox_mse_total)
# avg_vox2vox_r2 = sum(vox2vox_r2_total) / len(vox2vox_r2_total)

# print(f'Avg MSE for DL: {avg_dl_mse}')
# print(f'Avg MSE for CorrField: {avg_corr_mse}')
# print(f'Avg MSE for Vox2Vox: {avg_vox2vox_mse}\n')

# print(f'Avg R^2 for DL: {avg_dl_r2}')
# print(f'Avg R^2 for CorrField: {avg_corr_r2}')
# print(f'Avg R^2 for Vox2Vox: {avg_vox2vox_r2}')

# fig, ax = plt.subplots(1, figsize=(8, 6))
# ax.boxplot([corr_mse_total, dl_mse_total, vox2vox_mse_total], labels=['CorrField', 'DL', 'Vox2Vox'])
# ax.set_title('MSE Distribution')
# ax.set_ylabel('MSE')
# ax.set_xlabel('Method')

# plt.savefig('mse_distribution_5samples_val_proper_dl_vox2vox_60e.png')
# plt.close()

# fig, ax = plt.subplots(1, figsize=(8, 6))
# ax.boxplot([corr_r2_total, dl_r2_total, vox2vox_r2_total], labels=['CorrField', 'DL', 'Vox2Vox'])
# ax.set_title('R^2 Distribution')
# ax.set_ylabel('R^2')
# ax.set_xlabel('Method')

# plt.savefig('r2_distribution_5samples_val_proper_dl_vox2vox_60e.png')
# plt.close()