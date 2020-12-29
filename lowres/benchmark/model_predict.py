import time
from os.path import join as jp
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from dpipe.predict.shape import add_extract_dims, patches_grid, divisible_shape
from dpipe.predict.functional import preprocess
from dpipe.medim.shape_ops import pad
from dpipe.io import load
from dpipe.torch.model import inference_step

from lowres.torch.model import inference_step_lowres
from lowres.utils import get_lowres_dir_name
from lowres.path import luna_data_path


def get_predict(_model_name, _ps):
    n_chans_in = n_chans_out = 1
    model_path = jp(get_lowres_dir_name(), f'../model/{_model_name}.pth')

    print(f'Loading the {_model_name} model weights...', end=' ')

    if 'deepmedic39' == _model_name:
        from lowres.model.deepmedic39 import get_dm39
        architecture = get_dm39(n_chans_in=n_chans_in, n_chans_out=n_chans_out)

        @divisible_shape(divisor=[3] * 3, padding_values=np.min)
        @preprocess(pad, padding=[[24] * 2] * 3, padding_values=np.min)
        def model_predict(_x):
            return inference_step(_x, architecture=architecture, activation=torch.sigmoid)

        stride = 9

    elif 'enet3d' == _model_name:
        from lowres.model.enet3d import ENet3D
        architecture = ENet3D(n_chans_in=n_chans_in, n_chans_out=n_chans_out)

        @divisible_shape(divisor=[8] * 3, padding_values=np.min)
        def model_predict(_x):
            return inference_step(_x, architecture=architecture, activation=torch.sigmoid)

        stride = 8

    elif 'mobilenetv2unet3d' == _model_name:
        from lowres.model.mobilenetv2unet3d.mobilenetv2unet3d import MobileNetV2UNet3D
        architecture = MobileNetV2UNet3D(n_chans_in=n_chans_in, n_chans_out=n_chans_out)

        @divisible_shape(divisor=[32] * 3, padding_values=np.min)
        def model_predict(_x):
            return inference_step(_x, architecture=architecture, activation=torch.sigmoid)

        stride = 8

    elif 'unet3d' == _model_name:
        from lowres.model.unet3d import UNet3D
        architecture = UNet3D(n_chans_in=n_chans_in, n_chans_out=n_chans_out)

        @divisible_shape(divisor=[8] * 3, padding_values=np.min)
        def model_predict(_x):
            return inference_step(_x, architecture=architecture, activation=torch.sigmoid)

        stride = 8

    elif 'lowres' == _model_name:
        from lowres.model.lowres.modelx8 import ModelX8, SCALE_FACTOR
        from lowres.model.lowres.lowres import LowRes
        n_features = 16
        lowres_margin = 1
        model_x8 = ModelX8(n_chans_in=n_chans_in, n_chans_out=n_chans_out, n_features=n_features)
        model_lowres = LowRes(n_chans_in=n_chans_in, n_chans_out=n_chans_out, n_features=n_features)
        architecture = nn.ModuleList((model_x8, model_lowres))

        @divisible_shape(divisor=[SCALE_FACTOR] * 3, padding_values=np.min)
        def model_predict(_x):
            return inference_step_lowres(_x, model_x8=model_x8, model_lowres=model_lowres, activation=torch.sigmoid,
                                         margin=lowres_margin)

        stride = 8

    else:
        raise ValueError(
            '`_model_name` should be in (`deepmedic39`, `enet3d`, `mobilenetv2unet3d`, `unet3d`, `lowres`),'
            f' `{_model_name}` is given.')

    architecture.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    patch_size = np.array([_ps] * 3)
    patch_stride = np.array([_ps - stride] * 3)

    @add_extract_dims()
    @add_extract_dims()
    @patches_grid(patch_size, patch_stride)
    def _predict(_x):
        return model_predict(_x)

    print(f'Done')

    return _predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--ps', required=True, type=int)
    args = parser.parse_known_args()[0]
    ps = args.ps
    model_name = args.model_name

    meta_name = 'metadata.csv'
    data_path = luna_data_path
    meta_path = jp(data_path, f'{meta_name}')
    df = pd.read_csv(meta_path, index_col='id')

    predict = get_predict(_model_name=model_name, _ps=ps)

    print('Loading the scans...', end=' ')
    xs = []
    for _id in df.index:
        xs.append(load(jp(data_path, df.loc[_id]['image'])))
    print('Done')

    print('Running the benchmark...')
    predict_time = []
    for x in tqdm(xs):
        start_time = time.time()
        predict(x)
        finish_time = time.time()

        predict_time.append(finish_time - start_time)

    avg_time = np.mean(predict_time)
    std_time = np.std(predict_time)

    print(f'\nCPU time: {avg_time:.1f} +- {std_time:.1f} sec')
