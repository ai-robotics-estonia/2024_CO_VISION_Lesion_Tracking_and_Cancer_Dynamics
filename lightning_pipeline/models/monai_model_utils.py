from omegaconf import OmegaConf
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector


def init_retinanet_detector(
    detector: RetinaNetDetector,
    atss_matcher_params: OmegaConf = None,
    hard_negative_sampler_params: OmegaConf = None,
    target_keys_params: OmegaConf = None,
    box_selector_params: OmegaConf = None,
    sliding_window_inferer_params: OmegaConf = None,
):
    detector.set_atss_matcher(**atss_matcher_params)
    detector.set_hard_negative_sampler(**hard_negative_sampler_params)
    detector.set_target_keys(**target_keys_params)
    detector.set_box_selector_parameters(**box_selector_params)
    detector.set_sliding_window_inferer(**sliding_window_inferer_params)

    return detector
