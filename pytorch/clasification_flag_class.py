from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Union, List


@dataclass
class Flags:
    # General
    debug: bool = True
    outdir: str = "results/det"
    device: str = "cuda:0"

    # Data config
    imgdir_name: str = "vinbigdata-chest-xray-resized-png-256x256"
    # split_mode: str = "all_train"  # all_train or valid20
    seed: int = 111
    target_fold: int = 0  # 0~4
    label_smoothing: float = 0.0
    # Model config
    model_name: str = "resnet18"
    model_mode: str = "normal"  # normal, cnn_fixed supported
    # Training config
    epoch: int = 20
    batchsize: int = 8
    valid_batchsize: int = 16
    num_workers: int = 4
    snapshot_freq: int = 5
    ema_decay: float = 0.999  # negative value is to inactivate ema.
    scheduler_type: str = ""
    scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    scheduler_trigger: List[Union[int, str]] = field(default_factory=lambda: [1, "iteration"])
    aug_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})
    mixup_prob: float = -1.0  # Apply mixup augmentation when positive value is set.

    def update(self, param_dict: Dict) -> "Flags":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self
    
flags_template_dict = {
    "debug": False,  # Change to True for fast debug run!
    "outdir": "results/tmp_debug",
    # Data
    "imgdir_name": "vinbigdata-chest-xray-resized-png-256x256",
    # Model
    "model_name": "resnet18",
    # Training
    "num_workers": 4,
    "epoch": 15,
    "batchsize": 8,
    "scheduler_type": "CosineAnnealingWarmRestarts",
    "scheduler_kwargs": {"T_0": 28125},  # 15000 * 15 epoch // (batchsize=8)
    "scheduler_trigger": [1, "iteration"],
    "aug_kwargs": {
        "HorizontalFlip": {"p": 0.5},
        "ShiftScaleRotate": {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5},
        "RandomBrightnessContrast": {"p": 0.5},
        "CoarseDropout": {"max_holes": 8, "max_height": 25, "max_width": 25, "p": 0.5},
        "Blur": {"blur_limit": [3, 7], "p": 0.5},
        "Downscale": {"scale_min": 0.25, "scale_max": 0.9, "p": 0.3},
        "RandomGamma": {"gamma_limit": [80, 120], "p": 0.6},
    }
}