from dpipe.io import load, load_json

from connectome import Source, meta

from .utils import LATEST_COMMIT, glob, read


class LUNA(Source):
    _image = _lungs_mask = _lung_nodules_mask = LATEST_COMMIT

    @meta
    def ids(_image):
        folders = glob('image/*', version=_image)
        return tuple(sorted(i.name for i in folders))

    def origin(i, _image):
        return read(load_json, f'image/{i}/origin.json', version=_image)

    def spacing(i, _image):
        return read(load_json, f'image/{i}/spacing.json', version=_image)

    def shape(i, _image):
        return read(load_json, f'image/{i}/shape.json', version=_image)

    def subset(i, _image):
        return int(read(load_json, f'image/{i}/subset.json', version=_image)[-1])

    def image(i, _image):
        return read(load, f'image/{i}/image.npy.gz', version=_image, ext='.npy.gz')

    def lungs_mask(i, _lungs_mask):
        return read(load, f'lungs_mask/{i}.npy.gz', version=_lungs_mask, ext='.npy.gz')

    def lung_nodules_mask(i, _lung_nodules_mask):
        return read(load, f'lung_nodules_mask/{i}.npy.gz', version=_lung_nodules_mask, ext='.npy.gz')


def get_n_tumors(ids, df):
    return df['n_tumors'].loc[ids].values
