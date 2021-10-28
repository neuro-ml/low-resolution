import numpy as np
import cv2

from skimage import measure


def get_connected_components(y):
    cc = measure.label(y, neighbors=8)
    return np.array(cc, dtype='float32')


def interpolate_np(x, scale_factor, axes=(-1, -2, -3)):
    for ax in axes:
        x = np.repeat(x, scale_factor, axis=ax)
    return x
