import numpy as np
from skimage.metrics import structural_similarity


def compute_ssim(original, reconstructed, data_range=255, channel_axis=-1):
    if original.shape != reconstructed.shape:
        raise ValueError("Images must have the same shape")
    return structural_similarity(
        original,
        reconstructed,
        data_range=data_range,
        channel_axis=channel_axis,
    )
