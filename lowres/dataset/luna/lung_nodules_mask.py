########################################################################
# source: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
########################################################################

import argparse
from pathlib import Path

from tqdm import tqdm

import numpy as np
import pandas as pd

from dpipe.io import load, save

from lowres.dataset.luna.utils import get_nodules, nodules2centers, center2hit, fill3d


REVERSE_IDS = [
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.282512043257574309474415322775",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.801945620899034889998809817499",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.943403138251347598519939390311",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.964952370561266624992539111877",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.144883090372691745980459537053",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.127965161564033605177803085629",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.148447286464082095534651426689",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.219349715895470349269596532320",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.123697637451437522065941162930",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.177252583002664900748714851615",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.312127933722985204808706697221"
]


def nodules2target(expert_nodules, rel_centers, origin, spacing, shape, r=10) -> np.ndarray:
    """Builds segmentation mask from the given expert delineation."""

    lung_nodules_mask = np.zeros(shape, dtype='float32')

    for nodules in expert_nodules:
        centers = nodules2centers(nodules, origin[-1], spacing[-1])

        for center, nodule in zip(centers, nodules):
            if center2hit(center, rel_centers, r=r):
                lung_nodules_mask += fill3d(np.zeros(shape, dtype='float32'), nodule, origin[-1], spacing[-1])

    n_experts = 4

    return (lung_nodules_mask / n_experts) >= .5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('source', type=str, help='Absolute path to the downloaded LUNA16 dataset.')

    args = parser.parse_args()

    source = Path(args.source)
    dest = source.parent / 'LUNA16_processed' / 'lung_nodules_mask'
    dest.mkdir(exist_ok=True, parents=True)

    nodules_df = pd.read_csv(source / 'annotations.csv')

    for series_uid, nodules in tqdm(nodules_df.groupby('seriesuid')):
        if (dest / f'{series_uid}.npy.gz').exists():
            continue

        abs_coords = nodules[['coordX', 'coordY', 'coordZ']].values

        try:
            chars = ('origin', 'spacing', 'shape')
            origin, spacing, shape = map(lambda x: np.array(load(dest.parent / f'image/{series_uid}/{x}.json')), chars)

            if series_uid in REVERSE_IDS:
                abs_coords[:, :2] = 2 * origin[:2] - abs_coords[:, :2]

            rel_coords = list(map(lambda x: np.int64(np.round((x - origin) / spacing)), abs_coords))

            expert_nodules = get_nodules(series_uid, source / 'tcia-lidc-xml')
            lung_nodules_mask = nodules2target(expert_nodules, rel_coords, origin, spacing, shape)

            save(lung_nodules_mask, dest / f'{series_uid}.npy.gz', compression=1, timestamp=0)
        except Exception as e:
            print(f'Adding lung nodules mask {series_uid} failed with {e.__class__.__name__}: {str(e)}.')
