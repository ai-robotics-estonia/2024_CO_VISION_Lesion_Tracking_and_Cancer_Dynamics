import os
import json
import numpy as np
import torch

from monai.data.utils import list_data_collate

# --------------------- NLST Dataset ---------------------

def get_files(data_dir):
    """
    Get L2R train/val files from NLST challenge
    """
    data_json = os.path.join(data_dir, "NLST_dataset.json")

    with open(data_json) as file:
        data = json.load(file)

    train_files = []
    for pair in data["training_paired_images"]:
        nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
        nam_moving = os.path.basename(pair["moving"]).split(".")[0]
        train_files.append(
            {
                "fixed_image": os.path.join(data_dir, "imagesTr", nam_fixed + ".nii.gz"),
                "moving_image": os.path.join(data_dir, "imagesTr", nam_moving + ".nii.gz"),
                "fixed_label": os.path.join(data_dir, "masksTr", nam_fixed + ".nii.gz"),
                "moving_label": os.path.join(data_dir, "masksTr", nam_moving + ".nii.gz"),
                "fixed_keypoints": os.path.join(data_dir, "keypointsTr", nam_fixed + ".csv"),
                "moving_keypoints": os.path.join(data_dir, "keypointsTr", nam_moving + ".csv"),
            }
        )

    val_files = []
    for pair in data["registration_val"]:
        nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
        nam_moving = os.path.basename(pair["moving"]).split(".")[0]
        val_files.append(
            {
                "fixed_image": os.path.join(data_dir, "imagesTr", nam_fixed + ".nii.gz"),
                "moving_image": os.path.join(data_dir, "imagesTr", nam_moving + ".nii.gz"),
                "fixed_label": os.path.join(data_dir, "masksTr", nam_fixed + ".nii.gz"),
                "moving_label": os.path.join(data_dir, "masksTr", nam_moving + ".nii.gz"),
                "fixed_keypoints": os.path.join(data_dir, "keypointsTr", nam_fixed + ".csv"),
                "moving_keypoints": os.path.join(data_dir, "keypointsTr", nam_moving + ".csv"),
            }
        )

    return train_files, val_files



# --------------------- TUH Dataset ---------------------

# def split_train_val(dict: dict, val_ratio: int = 0.1) -> dict:
#     """
#     Split the dataset into train and validation sets.
#     """
#     keys = list(dict.keys())
#     val_size = int(len(keys) * val_ratio)
#     val_keys = set(keys[:val_size])

#     train_dict = {k: v for k, v in dict.items() if k not in val_keys}
#     val_dict = {k: v for k, v in dict.items() if k in val_keys}

#     return train_dict, val_dict

# def get_files(data_dir: str) -> tuple:
#     with open(data_dir) as file:
#         data = json.load(file)
    
#     train_dict, val_dict = split_train_val(data, val_ratio=0.1)

#     train_files, val_files = [], []
#     for pid, info in train_dict.items():
#         train_files.append(
#             {
#                 "fixed_image": info['baseline']['path_to_nifti'],
#                 "moving_image": info['followup']['path_to_nifti'],
#                 "fixed_label": info['baseline']['path_to_lung_mask'],
#                 "moving_label": info['followup']['path_to_lung_mask'],
#                 "fixed_keypoints": info["baseline"]["path_to_keypoints"],
#                 "moving_keypoints": info["followup"]["path_to_keypoints"],
#             }
#         )

#     for pid, info in val_dict.items():
#         val_files.append(
#             {
#                 "fixed_image": info['baseline']['path_to_nifti'],
#                 "moving_image": info['followup']['path_to_nifti'],
#                 "fixed_label": info['baseline']['path_to_lung_mask'],
#                 "moving_label": info['followup']['path_to_lung_mask'],
#                 "fixed_keypoints": info["baseline"]["path_to_keypoints"],
#                 "moving_keypoints": info["followup"]["path_to_keypoints"],
#             }
#         )

#     return train_files, val_files


def collate_fn(batch):
    """
    Custom collate function.
    Some background:
        Collation is the "collapsing" of a list of N-dimensional tensors into a single (N+1)-dimensional tensor.
        The `Dataloader` object  performs this step after receiving a batch of (transformed) data from the
        `Dataset` object.
        Note that the `Resized` transform above resamples all image tensors to a shape `spatial_size`,
        thus images can be easily collated.
        Keypoints, however, are of different row-size and thus cannot be easily collated
        (a.k.a. "ragged" or "jagged" tensors): [(n_0, 3), (n_1, 3), ...]
        This function aligns the row-size of these tensors such that they can be collated like
        any regular list of tensors.
        To do this, the max number of keypoints is determined, and shorter keypoint-lists are filled up with NaNs.
        Then, the average-TRE loss below can be computed via `nanmean` aggregation (i.e. ignoring filled-up elements).
    """
    max_length = 0
    for data in batch:
        length = data["fixed_keypoints"].shape[0]
        if length > max_length:
            max_length = length
    for data in batch:
        length = data["fixed_keypoints"].shape[0]
        data["fixed_keypoints"] = torch.concat(
            (data["fixed_keypoints"], float("nan") * torch.ones((max_length - length, 3))), dim=0
        )
        data["moving_keypoints"] = torch.concat(
            (data["moving_keypoints"], float("nan") * torch.ones((max_length - length, 3))), dim=0
        )
    # note monai data usage
    return list_data_collate(batch)

