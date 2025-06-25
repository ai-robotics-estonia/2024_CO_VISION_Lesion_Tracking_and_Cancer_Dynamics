import pandas as pd
import torch

from monai.transforms import MapTransform


class LoadKeypointsd(MapTransform):
    """
    Load keypoints from csv file
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            keypoints = d[key]
            keypoints = pd.read_csv(keypoints, header=None)
            keypoints = keypoints.to_numpy()
            keypoints = torch.as_tensor(keypoints)
            d[key] = keypoints  # [N, 3]
        return d


# need to print results and check affine and affine original
# some transforms change affine matrix
# in my version I still neded to work with nii files, numpy will not have affine matrix
# in transforms pipeline all transforms are first applier ti image and then to keypoints

class TransformKeypointsd(MapTransform):
    """
    Applies any potential linear image transformation to keypoint values
    """

    def __init__(self, keys_keypoints, keys_images, ras=False):
        # super.__init__(self)
        self.keys_keypoints = keys_keypoints
        self.keys_images = keys_images
        # default is actually used in code below
        self.ras = ras

    def __call__(self, data):
        d = dict(data)
        for kp, ki in zip(self.keys_keypoints, self.keys_images):
            # Get image meta data
            image = d[ki]
            meta = image.meta
            # Get keypoints
            keypoints_ijk = d[kp]
            # Get transformation (in voxel space)
            affine = meta["affine"]
            original_affine = torch.as_tensor(meta["original_affine"], dtype=affine.dtype, device=affine.device)
            transforms_affine = (
                original_affine.inverse() @ affine
            )  # Assumes: affine = original_affine @ transforms_affine
            # why inverse here?
            transforms_affine = transforms_affine.inverse()
            if self.ras:
                # RAS space
                transforms_affine = original_affine @ transforms_affine
            # Apply transformation to keypoints
            keypoints_ijk_moved = torch.cat((keypoints_ijk, torch.ones((keypoints_ijk.shape[0]), 1)), dim=1)
            keypoints_ijk_moved = (transforms_affine @ keypoints_ijk_moved.T).T
            keypoints_ijk_moved = keypoints_ijk_moved[:, :3]
            keypoints_ijk_moved = keypoints_ijk_moved.float()

            d[kp] = keypoints_ijk_moved  # [N, 3]

        return d