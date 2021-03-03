



from typing import Dict

import albumentations as A


class Transform:
        '''aug_kwargs:
    HorizontalFlip: {"p": 0.5}
    ShiftScaleRotate: {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5}
    RandomBrightnessContrast: {"p": 0.5}
    CoarseDropout: {"max_holes": 8, "max_height": 25, "max_width": 25, "p": 0.5}
    Blur: {"blur_limit": [3, 7], "p": 0.5}
    Downscale: {"scale_min": 0.25, "scale_max": 0.9, "p": 0.3}
    RandomGamma: {"gamma_limit": [80, 120], "p": 0.6}
    '''
    def __init__(self, aug_kwargs: Dict):
        self.transform = A.Compose(
            [getattr(A, name)(**kwargs) for name, kwargs in aug_kwargs.items()]
        )

    def __call__(self, image):
        image = self.transform(image=image)["image"]
        return image