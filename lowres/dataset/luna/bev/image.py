import argparse
from pathlib import Path

from tqdm import trange, tqdm

import SimpleITK

import numpy as np

from dpipe.io import save


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('source', type=str, help='Absolute path to the downloaded LUNA16 dataset.')

    args = parser.parse_args()

    source = Path(args.source)
    dest = source.parent / 'LUNA16_processed' / 'image'
    dest.mkdir(exist_ok=True, parents=True)

    for i in trange(10):
        for series_path in tqdm((source / f'subset{i}').glob('*.mhd')):
            series_uid = series_path.name[:-4]

            if (dest / series_uid).exists():
                continue

            try:
                itk_image = SimpleITK.ReadImage(str(series_path))

                image = np.swapaxes(SimpleITK.GetArrayFromImage(itk_image), 0, 2)
                origin, spacing = map(np.array, (itk_image.GetOrigin(), itk_image.GetSpacing()))

                save_path = dest / series_uid
                save_path.mkdir()

                save(image, save_path / f'image.npy.gz', compression=1, timestamp=0)
                save(origin, save_path / 'origin.json')
                save(image.shape, save_path / 'shape.json')
                save(spacing, save_path / 'spacing.json')
                save(f'subset{i}', save_path / 'subset.json')
            except Exception as e:
                print(f'Adding image {series_uid} failed with {e.__class__.__name__}: {str(e)}.')
