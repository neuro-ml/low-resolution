import random
from itertools import islice

import numpy as np

from dpipe.medim.itertools import make_chunks
from dpipe.medim.patch import sample_box_center_uniformly
from dpipe.medim.box import get_centered_box
from dpipe.medim.shape_ops import crop_to_box
from dpipe.batch_iter import combine_to_arrays

SPATIAL_DIMS = (-3, -2, -1)


def infinite(source, *transformers, batch_size, batches_per_epoch, combiner=combine_to_arrays):
    def pipeline():
        for entry in source:
            for transformer in transformers:
                entry = transformer(entry)
            yield entry

    return lambda: islice(map(combiner, make_chunks(pipeline(), batch_size, False)), batches_per_epoch)


def sample_center_uniformly(shape, patch_size, spatial_dims):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size)
    else:
        return spatial_shape // 2


def center_choice(inputs, y_patch_size, nonzero_fraction=0.5, tumor_sampling=True):
    x, y, centers = inputs

    y_patch_size = np.array(y_patch_size)
    if len(centers) > 0 and np.random.uniform() < nonzero_fraction:
        center = random.choice(random.choice(centers)) if tumor_sampling else random.choice(centers)
    else:
        center = sample_center_uniformly(y.shape, patch_size=y_patch_size, spatial_dims=SPATIAL_DIMS)

    return x, y, center


def extract_patch(inputs, x_patch_size, y_patch_size):
    x, y, center = inputs

    x_patch_size = np.array(x_patch_size)
    y_patch_size = np.array(y_patch_size)
    x_spatial_box = get_centered_box(center, x_patch_size)
    y_spatial_box = get_centered_box(center, y_patch_size)

    x_patch = crop_to_box(x, box=x_spatial_box, axes=SPATIAL_DIMS, padding_values=np.min(x))
    y_patch = crop_to_box(y, box=y_spatial_box, axes=SPATIAL_DIMS, padding_values=0)
    return x_patch, y_patch
