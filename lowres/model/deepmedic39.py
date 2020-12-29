from functools import partial

import torch.nn as nn

import dpipe.layers as lrs


def get_dm39(n_chans_in, n_chans_out):
    init_n_chans = 30
    structure = [init_n_chans, 40, 40, 50]

    rb = partial(lrs.ResBlock3d, kernel_size=3)

    path1 = nn.Sequential(
        lrs.CenteredCrop([16, 16, 16]),
        lrs.PostActivation3d(n_chans_in, init_n_chans, kernel_size=3),
        nn.Conv3d(init_n_chans, init_n_chans, kernel_size=3, bias=False),
        lrs.make_consistent_seq(layer=rb, channels=structure),
    )

    path2 = nn.Sequential(
        nn.AvgPool3d(kernel_size=3),
        lrs.PostActivation3d(n_chans_in, init_n_chans, kernel_size=3),
        nn.Conv3d(init_n_chans, init_n_chans, kernel_size=3, bias=False),
        lrs.make_consistent_seq(layer=rb, channels=structure),
        nn.Upsample(scale_factor=3)
    )

    architecture = nn.Sequential(
        lrs.SplitCat(path1, path2),
        rb(100, 150, kernel_size=1),
        lrs.PreActivation3d(150, n_chans_out, kernel_size=1, bias=False),
        nn.BatchNorm3d(n_chans_out),
    )

    return architecture
