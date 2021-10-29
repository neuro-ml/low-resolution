import argparse
from pathlib import Path

from tqdm import tqdm

import SimpleITK

import numpy as np

from dpipe.io import save


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('source', type=str, help='Absolute path to the downloaded LUNA16 dataset.')

    args = parser.parse_args()

    source = Path(args.source)
    dest = source.parent / 'LUNA16_processed' / 'lungs_mask'
    dest.mkdir(exist_ok=True, parents=True)

    for series_path in tqdm((source / 'seg-lungs-LUNA16').glob('*.mhd')):
        series_uid = series_path.name[:-4]

        if (dest / f'{series_uid}.npy.gz').exists():
            continue

        try:
            itk_lungs_mask = SimpleITK.ReadImage(str(series_path))
            lungs_mask = np.swapaxes(SimpleITK.GetArrayFromImage(itk_lungs_mask), 0, 2)
            save(lungs_mask, dest / f'{series_uid}.npy.gz', compression=1, timestamp=0)
        except Exception as e:
            print(f'Adding lungs_mask {series_uid} failed with {e.__class__.__name__}: {str(e)}.')
