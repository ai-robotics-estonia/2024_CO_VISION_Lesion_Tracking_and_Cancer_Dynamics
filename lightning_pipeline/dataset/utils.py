import cv2
import torch
import numpy as np
from scipy.spatial.distance import cdist


def generate_gauss_kernel(size: int = 3) -> np.ndarray:
    kernel = cv2.getGaussianKernel(size, 0)
    kernel = kernel * kernel.T
    kernel /= np.max(kernel)
    return kernel

def generate_cross(size: int = 3) -> np.ndarray:
    # TODO: implement for other cross sizes
    cross = [[5, 1, 5],
             [1, 7, 1],
             [5, 1, 5]]
    return np.array(cross)

def find_nearest(array, thr):
    # find nearest value >= thr
    return np.min(array[array >= thr])

def center_distance_based(
        mask: np.ndarray, 
        mode: str, 
        thr_mask: float = 0.5, 
        thr_distance: float = 0.5, 
        prob_jitter: float = 0.5
    ) -> np.ndarray:

    mask_thr = mask >= thr_mask
    dist = cv2.distanceTransform(mask_thr.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    if (mode == "train") and (np.random.rand() < prob_jitter):
        mask_candidates = dist >= thr_distance * dist.max()
    elif mode == "test":
        mask_candidates = dist == find_nearest(dist, thr_distance * dist.max())
        np.random.seed(42)
    else:
        mask_candidates = dist == dist.max()   # [y, x]
    
    candidates_coords = np.argwhere(mask_candidates)
    center = candidates_coords[np.random.randint(candidates_coords.shape[0], size=1), :]
    
    return center


def opposite_points(
        mask: np.ndarray,
        mode: str,
        thr_mask: float = 0.5,
        prob_random_pair: float = 0.5,
        prob_jitter: float = 0.25
    ) -> np.ndarray:
    """Returns randomly selected 2 opposite points from the mask's contour"""

    if mode == "test":
        np.random.seed(42)

    _, cv2_thresh = cv2.threshold((255 * mask).astype(np.uint8), int(round(thr_mask * 255)), 255, 0)
    contours, _ = cv2.findContours(cv2_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) > 0:
        contour = max(contours, key = cv2.contourArea)
        contour = contour[:, 0]        # from cv2 format to adequate one
    else:
        return np.array([])
    
    hdist = cdist(contour, contour, metric='euclidean')
    
    # for mode = 'val' or 'test' we use longest axis only
    pair_idxs = np.unravel_index(hdist.argmax(), hdist.shape)

    if (mode in ["train", "test"]) and (np.random.rand() < prob_random_pair):
        # use random pair of points instead
        idx_0 = np.random.randint(len(contour))
        pair_idxs = [idx_0, np.argmax(hdist[idx_0])] 
        
    points = contour[pair_idxs, ]
    points = points[:, ::-1]   # [[x, y], ...] -> [[y, x], ...]

    # introduce point jittering
    if (mode in ["train", "test"]) and (np.random.rand() < prob_jitter):
        # calculate max bbox side (buit on these points)
        max_side = np.max(np.abs(points[0] - points[1]))

        delta_max = max_side // 4            # to avoid overlapping
        delta_max = min(delta_max, 3)       # force delta to be in range [0, 2] 
        
        if delta_max > 0:
            points += np.random.randint(-delta_max, delta_max, size=(2, 2))
            np.clip(points, 0, mask.shape[0]-1, out=points)

    return points


def draw_kernel(shape: np.ndarray, kernel: np.ndarray, points: np.ndarray) -> np.ndarray:

    kernel_size = kernel.shape[0]
    output_mask = np.zeros(shape).astype(float)

    # padding to avoid issues near the mask edges
    pad = kernel_size // 2

    output_mask = np.pad(
        output_mask,
        ((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0,
    )

    points = points + pad

    for [y, x] in points:
        output_mask[y - pad : y + pad + 1, x - pad : x + pad + 1] = np.maximum(
            kernel,
            output_mask[y - pad : y + pad + 1, x - pad : x + pad + 1]
        )

    # compensate padding
    output_mask = output_mask[pad: -pad, pad: -pad]

    return output_mask


def max_contour_mask(mask):
    max_contour_mask = np.zeros_like(mask)

    _, thresh = cv2.threshold((255 * mask).astype(np.uint8), 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        max_contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(max_contour_mask, [max_contour], -1, 1, -1)

    return mask * max_contour_mask

def prepare_deep_sv_masks(mask: torch.Tensor, stages: int) -> list[torch.Tensor]:
    dim_pool_dict = {
        3: torch.nn.MaxPool2d(2, stride=2),
        4: torch.nn.MaxPool3d(2, stride=2)
    }
    max_pool = dim_pool_dict[mask.dim()]
    
    masks_list = [mask]
    for _ in range(stages - 1):
        masks_list.append(max_pool(masks_list[-1]))

    return masks_list