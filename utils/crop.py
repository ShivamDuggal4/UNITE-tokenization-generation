"""
Center crop utilities matching JiT / ADM preprocessing.
Reference: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
"""

import numpy as np
from PIL import Image


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM / JiT.

    1. Iteratively downscales by 2x using BOX filter while possible
    2. Resizes shortest edge to image_size using BICUBIC
    3. Center crops to image_size x image_size
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class CenterCropTransform:
    """Picklable transform wrapper for center_crop_arr (must be at module level)."""
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return center_crop_arr(img, self.size)
