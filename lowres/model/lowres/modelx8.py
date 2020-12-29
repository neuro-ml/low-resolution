import torch.nn as nn

from dpipe.layers.conv import PreActivation3d
from dpipe.layers.resblock import ResBlock3d

SCALE_FACTOR = 8


class ModelX8(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, n_features=32):
        super().__init__()
        self.y_downsampling = SCALE_FACTOR

        self.x1_to_x2 = nn.AvgPool3d(kernel_size=2, ceil_mode=True)

        self.path_x2 = nn.Sequential(
            nn.Conv3d(n_chans_in, n_features, kernel_size=3, padding=1, bias=False),
            PreActivation3d(n_features, n_features, kernel_size=3, padding=1),
            ResBlock3d(n_features, n_features, kernel_size=3, padding=1)
        )

        self.path_x4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, ceil_mode=True),
            PreActivation3d(n_features, n_features * 2, kernel_size=3, padding=1),
            ResBlock3d(n_features * 2, n_features * 2, kernel_size=3, padding=1),
            ResBlock3d(n_features * 2, n_features * 2, kernel_size=3, padding=1)
        )

        self.path_x8 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, ceil_mode=True),
            PreActivation3d(n_features * 2, n_features * 4, kernel_size=3, padding=1),
            ResBlock3d(n_features * 4, n_features * 4, kernel_size=3, padding=1),
            ResBlock3d(n_features * 4, n_features * 4, kernel_size=3, padding=1),
            ResBlock3d(n_features * 4, n_features * 4, kernel_size=1, padding=0)
        )

        self.output_x8 = nn.Sequential(
            PreActivation3d(n_features * 4, n_chans_out, kernel_size=1, bias=False),
            nn.BatchNorm3d(n_chans_out)
        )

    def forward_features(self, x):
        return self.forward(x, return_features=True)

    def forward(self, x, return_features=False):
        x2_inp = self.x1_to_x2(x)

        x2_f = self.path_x2(x2_inp)
        x4_f = self.path_x4(x2_f)
        x8_f = self.path_x8(x4_f)

        if return_features:
            return [x, x2_inp, x2_f, x4_f, x8_f, self.output_x8(x8_f)]
        else:
            return self.output_x8(x8_f)
