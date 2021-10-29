import numpy as np

from skimage.measure import label
from skimage.morphology import binary_dilation, binary_erosion

from connectome import Transform, positional

from dpipe.im import crop_to_box
from dpipe.im.box import mask2bounding_box, add_margin, limit_box
from dpipe.itertools import collect


class DropTracheaMask(Transform):
    __inherit__ = True

    def lungs_mask(lungs_mask):
        labels, volumes = np.unique(lungs_mask, return_counts=True)
        trachea_label = labels[np.argmin(volumes)]
        lungs_mask[lungs_mask == trachea_label] = 0

        return lungs_mask


class ProcessLungsMask(Transform):
    __inherit__ = True

    def lungs_mask(lungs_mask):
        def _mask2closed_mask(mask, margin=1):

            if margin == 0:
                return lungs_mask

            for _ in range(margin):
                mask = binary_dilation(mask)

            for _ in range(margin):
                mask = binary_erosion(mask)

            return mask.astype('int32')

        def _mask2dilated_mask(mask, margin=1):

            if margin == 0:
                return mask

            for _ in range(margin):
                mask = binary_dilation(mask)

            return mask.astype('int32')

        return _mask2dilated_mask(_mask2closed_mask(lungs_mask, margin=10), margin=3)


class CropToLungsMask(Transform):
    __inherit__ = True

    def _bbox(lungs_mask, shape):
        return limit_box(add_margin(mask2bounding_box(lungs_mask), margin=5), shape)

    @positional
    def image(x, _bbox):
        return crop_to_box(x, _bbox)

    lungs_mask = lung_nodules_mask = image


class ApplyLungsMask(Transform):
    __inherit__ = True

    def image(image, lungs_mask):
        lungs_mask = lungs_mask > 0
        if np.sum(lungs_mask) == 0:
            raise ValueError('The obtained mask is empty')

        image *= lungs_mask
        image[lungs_mask == 0] = np.min(image)

        return image


class MakeCentersNComponents(Transform):
    __inherit__ = True

    def _cc(lung_nodules_mask):
        assert np.issubdtype(lung_nodules_mask.dtype, bool)

        return np.array(label(lung_nodules_mask, connectivity=3), dtype='float32')

    cc = _cc

    @collect
    def lung_nodules_centers(_cc):
        for l in range(1, int(_cc.max() + 1)):
            yield np.argwhere(_cc == l)

    def n_lung_nodules(_cc):
        return int(_cc.max())


def normalize_intensities(image, hu_window=(-1000, 300)):
    min_hu, max_hu = hu_window

    return np.clip((image.astype(np.float32) - min_hu) / (max_hu - min_hu), 0, 1)
