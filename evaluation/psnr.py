import numpy as np
from skimage.metrics import peak_signal_noise_ratio


def compute_psnr(original, reconstructed, data_range=255):
    if original.shape != reconstructed.shape:
        raise ValueError("Images must have the same shape")
    return peak_signal_noise_ratio(original, reconstructed, data_range=data_range)
