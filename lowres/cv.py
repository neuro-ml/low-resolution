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


def fill3d(img3d, nodule, z_origin, z_spacing):
    """Fills LUNA target ``img3d`` with given ``nodule`` roi."""
    img3d = np.float32(img3d)

    for roi in nodule:
        z = int((roi[0] - z_origin) / z_spacing)
        pts = np.int32([roi[1]])
        img = np.zeros_like(img3d[..., z], dtype='float32').T

        img3d[::, ::, z] += cv2.fillPoly(img.copy(), pts, 1).T

    return np.clip(img3d, 0, 1)
