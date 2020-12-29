import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from lowres.model.mobilenetv2unet3d.mobilenetv2 import MobileNetV2, InvertedResidual


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv1 = InvertedResidual(in_ch + skip_ch, out_ch, stride=1, expand_ratio=6)
        self.conv2 = InvertedResidual(out_ch, out_ch, stride=1, expand_ratio=6)

    def forward(self, x, x_skip):
        x = self.up(x)
        x = torch.cat([x, x_skip], dim=1)
        return self.conv2(self.conv1(x))


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = InvertedResidual(in_ch, out_ch, stride=1, expand_ratio=1)
        self.bn = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class MobileNetV2UNet3D(nn.Module):
    def __init__(self, n_chans_in, n_chans_out):
        super(MobileNetV2UNet3D, self).__init__()

        self.backbone = MobileNetV2(n_chans_in)

        self.up1 = UpBlock(320, 96, 96)
        self.up2 = UpBlock(96, 32, 32)
        self.up3 = UpBlock(32, 24, 24)
        self.up4 = UpBlock(24, 16, 16)

        self.outconv = OutConv(in_ch=16, out_ch=n_chans_out)

    def forward(self, x):
        for n in range(0, 2):
            x = self.backbone.features[n](x)
        x1 = x

        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x

        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x

        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x

        for n in range(14, 18):
            x = self.backbone.features[n](x)
        x5 = x

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)

        x = interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

        return x
