import albumentations as albu
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import generate_gauss_kernel, generate_cross, draw_kernel, max_contour_mask, prepare_deep_sv_masks, center_distance_based, opposite_points
import cv2


class OneclickDataset(Dataset):
    def __init__(
        self,
        files: list,
        ndim: int = 2,
        mode: str = "train",
        transforms: Optional[albu.Compose] = None,
        add_input: dict = {},
        deep_sv_stages: int = 1
    ):
        self.files = files
        self.ndim = ndim
        self.mode = mode
        self.deep_sv_stages = deep_sv_stages

        self.transforms = transforms

        add_input_kernel_dict = {
            2: generate_gauss_kernel(),     # gaussians for 2D
            3: generate_cross()             # x-shaped crosses for 3D
        }

        prepare_additional_inputs_dict = {
            2: self.prepare_additional_inputs_2d,
            3: self.prepare_additional_inputs_3d
        }

        self.point_prompt_dict = {
            1: center_distance_based,
            2: opposite_points
        }

        self.prepare_additional_inputs = prepare_additional_inputs_dict[self.ndim]

        if add_input:
            self.add_input = dict(add_input.copy())
            self.point_prompt = self.add_input.pop("point_prompt", None)
            self.add_input_kernel = add_input_kernel_dict[ndim]
        else:
            self.point_prompt = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]
        mask_path = image_path.replace("images", "masks")

        image = np.load(image_path)
        mask = np.load(mask_path).astype(np.float32)

        if isinstance(self.transforms, albu.Compose):
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # MONAI augmentations
        else:
            data = {'image': image, 'mask': mask}
            augmented = self.transforms(data)
            image, mask = augmented['image'], augmented['mask']

            # workaroud because of under the hood MONAI transforms to tensor
            image = image.detach().cpu().numpy().squeeze() if isinstance(image, torch.Tensor) else image
            mask = mask.detach().cpu().numpy().squeeze() if isinstance(mask, torch.Tensor) else mask

        image, mask = self.prepare_additional_inputs(image, mask)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        masks = prepare_deep_sv_masks(mask=mask, stages=self.deep_sv_stages) if self.deep_sv_stages > 1 else mask
        
        return image, masks
    
    def prepare_additional_inputs_2d(self, image, mask):
        # sagittal and coronal slices could contain few masks
        mask = max_contour_mask(mask)

        if self.point_prompt > 0:
            points = self.point_prompt_dict[self.point_prompt](mask, mode=self.mode, **self.add_input)
            add_input = draw_kernel(shape=mask.shape, kernel=self.add_input_kernel, points=points)
            image = np.stack([image, add_input])
        else:
            image = image[np.newaxis, ...]

        mask = mask[np.newaxis, ...]
        
        return image, mask

    def prepare_additional_inputs_3d(self, image, mask):
        if self.point_prompt > 0:
            # draw additional input inplace (on top of the image slice with the largest 2D mask)
            if self.point_prompt == 3:
                # for 6 points additional input we need to draw mask of 3 slices total
                image = self.get_six_points_3D_add_input(mask, image)
            else:
                image = self.get_middle_slice_3D_add_input(mask, image)

        # HxWxC -> CxHxW
        image = np.moveaxis(image, -1, 0)[np.newaxis, ...]
        mask = np.moveaxis(mask, -1, 0)[np.newaxis, ...]

        return image, mask

    def get_middle_slice_3D_add_input(self, mask: np.ndarray, image: np.ndarray):
        # calculate mask area for each axial slice
        mask_area = np.sum(np.sum(mask, axis=0), axis=0)
        max_area_slice_igx = np.argmax(mask_area)

        # slice with the max mask area
        max_area_mask = mask[:, :, max_area_slice_igx].copy()

        # introduce dilation to avoid
        dilation_kernel = np.ones((3, 3), np.uint8)
        max_area_mask = cv2.dilate(max_area_mask, dilation_kernel, iterations=2)

        points = self.point_prompt_dict[self.point_prompt](mask=max_area_mask, mode=self.mode, **self.add_input)
        add_input = draw_kernel(shape=max_area_mask.shape, kernel=self.add_input_kernel, points=points)

        # just a dumb workaround to make draw_kernel() works with both gaussians and custom crosses
        # all values are positive (before the following transformation) to handle overlapping while drawing gaussins/crosses (previous step)
        # [[5, 1, 5],          [[ 4, -4,  4]
        #  [1, 7, 1],    ->     [-4,  8, -4]
        #  [5, 1, 5]]           [ 4, -4,  4]]
        image[:, :, max_area_slice_igx] = image[:, :, max_area_slice_igx] * (add_input == 0) + 2 * (add_input - 3) * (add_input != 0)
        return image

    def get_six_points_3D_add_input(self, mask: np.ndarray, image: np.ndarray):

        padding = np.array([2, 2, 1]) # x and y padding is bigger to avoid overlap with 2D cross
        mask_shape_2d = np.array(mask.shape[:2])

        # find 3D bounding box coordinates around mask
        mask_area_nonzero = np.array(np.nonzero(mask))

        if mask_area_nonzero.shape[1] == 0:
            # mask is empty
            return image

        # find coordinates of the top left and bottom right corners of the bounding box
        tl =  np.maximum(np.min(mask_area_nonzero, axis=1) - padding, 0)
        br =  np.minimum(np.max(mask_area_nonzero, axis=1) + padding, np.array(mask.shape)-1)

        # points for top and bottom slices
        edge_points = np.array([(tl[:2] + br[:2]) // 2])[:, ::-1]

        # points for middle slice in [y, x] format
        middle_slice = (tl[2] + br[2]) // 2
        middle_points = np.array([[(tl[1] + br[1]) // 2, tl[0]], [(tl[1] + br[1]) // 2, br[0]],
                                  [tl[1], (tl[0] + br[0]) // 2], [br[1], (tl[0] + br[0]) // 2]])

        #add small jitter to the middle points
        if (self.mode == 'train') and (np.random.rand() < self.add_input["prob_jitter"]):

            max_side = np.max(br - tl) # max side of the bounding box
            delta_max = max_side // 4            # to avoid overlapping
            delta_max = min(delta_max, 3)
            middle_points += np.random.randint(-delta_max, delta_max, size=middle_points.shape)
            edge_points += np.random.randint(-delta_max, delta_max, size=edge_points.shape)
            # make sure that points are inside the mask
            np.clip(middle_points, 0, mask_shape_2d-1, out=middle_points)
            np.clip(edge_points, 0, mask_shape_2d-1, out=edge_points)

        # add crosses for top and bottom slices
        add_input_edge = draw_kernel(shape=mask_shape_2d, kernel=self.add_input_kernel, points=edge_points)
        add_input_middle = draw_kernel(shape=mask_shape_2d, kernel=self.add_input_kernel, points=middle_points)

        # just a dumb workaround to make draw_kernel() works with both gaussians and custom crosses
        # all values are positive (before the following transformation) to handle overlapping while drawing gaussins/crosses (previous step)
        # [[5, 1, 5],          [[ 4, -4,  4]
        #  [1, 7, 1],    ->     [-4,  8, -4]
        #  [5, 1, 5]]           [ 4, -4,  4]]
        # add all 6 points to the image
        image[:, :, tl[2]] = image[:, :, tl[2]] * (add_input_edge == 0) + 2 * (add_input_edge - 3) * (add_input_edge != 0)
        image[:, :, br[2]] = image[:, :, br[2]] * (add_input_edge == 0) + 2 * (add_input_edge - 3) * (add_input_edge != 0)
        image[:, :, middle_slice] = image[:, :, middle_slice] * (add_input_middle == 0) + 2 * (add_input_middle - 3) * (add_input_middle != 0)
        return image
