import numpy as np
import torch
import torch.nn as nn
from skimage import measure

from dpipe.torch import to_var
from dpipe.medim.box import mask2bounding_box, box2slices, add_margin, limit_box

from .structural_blocks import PreActivation3dNoBN, ResBlock3dNoBN


class LowRes(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, y_downsampling=8, n_features=32):
        super().__init__()

        self.y_downsampling = y_downsampling

        self.downsample_x2 = nn.MaxPool3d(kernel_size=2, ceil_mode=True)
        self.upsample_x8 = nn.Upsample(scale_factor=y_downsampling)
        self.upsample_x4 = nn.Upsample(scale_factor=y_downsampling // 2)
        self.upsample_x2 = nn.Upsample(scale_factor=y_downsampling // 4)

        self.x8_ch = 4 * n_features
        self.x4_ch = 2 * n_features
        self.x2_inp_ch = 1
        self.x2_ch = n_features + self.x2_inp_ch
        self.x_ch = n_chans_in
        self.x1_ch = n_features

        self.path_x8 = PreActivation3dNoBN(self.x8_ch, self.x4_ch, kernel_size=1, padding=0, bias=False)
        self.x8_up_x4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.path_x4 = PreActivation3dNoBN(self.x4_ch, self.x2_ch, kernel_size=1, padding=0, bias=False)
        self.x4_up_x2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.path_x2 = nn.Sequential(
            ResBlock3dNoBN(n_channels=self.x2_ch),
            ResBlock3dNoBN(n_channels=self.x2_ch),
            PreActivation3dNoBN(self.x2_ch, self.x1_ch, bias=False)
        )
        self.x2_up_x1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.path_x1 = nn.Sequential(
            ResBlock3dNoBN(n_channels=self.x1_ch + self.x_ch),
            ResBlock3dNoBN(n_channels=self.x1_ch + self.x_ch),
            ResBlock3dNoBN(n_channels=self.x1_ch + self.x_ch, kernel_size=1, padding=0),
        )

        self.output_x1 = nn.Sequential(
            nn.Conv3d(self.x1_ch + self.x_ch, n_chans_out, kernel_size=1, bias=False)
        )

    def prepare_masks(self, y8_single, margin=0):
        y8_single = y8_single[None]  # restoring to 5-dims
        y_pred_single = self.upsample_x8(y8_single)

        y8_cpu = y8_single[0][0].data.cpu().numpy()  # reducing to 3-dims
        y8_bin_cpu = y8_cpu > 0
        y8_lbls_cpu = measure.label(y8_bin_cpu)
        labels = np.unique(y8_lbls_cpu)

        if len(labels) == 1 and labels[0] == 0:
            return y_pred_single, None, None
        elif len(labels) > 1:
            labels = labels[1:]

        masks = []
        shapes_3d = []
        for lbl in labels:
            y8_lbl_cpu = np.zeros_like(y8_lbls_cpu)
            y8_lbl_cpu[y8_lbls_cpu == lbl] = True

            box = limit_box(add_margin(mask2bounding_box(y8_lbl_cpu), margin=margin), limit=y8_lbl_cpu.shape)
            y8_lbl_cpu[box2slices(box)] = True

            y8_lbl = to_var(y8_lbl_cpu[None, None]).to(y8_single)  # restoring to 5-dims

            masks.append(y8_lbl)
            shapes_3d.append(box[1] - box[0])

        return y_pred_single, masks, shapes_3d

    def restore_shape(self, y8):
        y = self.upsample_x8(y8)
        return y

    def forward(self, x_features_single, y8_mask, y8_mask_shape):
        x, x2_inp, x2_f, x4_f, x8_f = x_features_single

        # selecting component with y8_mask
        x8_flat = x8_f.masked_select(y8_mask > 0)
        x8_f = x8_flat.reshape(
            (1, x8_f.shape[1], y8_mask_shape[0], y8_mask_shape[1], y8_mask_shape[2])
        )

        y4_mask = self.upsample_x2(y8_mask) > 0
        x4_flat = x4_f.masked_select(y4_mask)
        x4_f = x4_flat.reshape(
            (1, x4_f.shape[1], y8_mask_shape[0] * 2, y8_mask_shape[1] * 2, y8_mask_shape[2] * 2)
        )

        y2_mask = self.upsample_x4(y8_mask) > 0
        x2_flat = x2_f.masked_select(y2_mask)
        x2_f = x2_flat.reshape(
            (1, x2_f.shape[1], y8_mask_shape[0] * 4, y8_mask_shape[1] * 4, y8_mask_shape[2] * 4)
        )
        x2_inp_flat = x2_inp.masked_select(y2_mask)
        x2_inp = x2_inp_flat.reshape(
            (1, x2_inp.shape[1], y8_mask_shape[0] * 4, y8_mask_shape[1] * 4, y8_mask_shape[2] * 4)
        )

        y1_mask = self.upsample_x8(y8_mask) > 0
        x_flat = x.masked_select(y1_mask)
        x = x_flat.reshape(
            (1, x.shape[1], y8_mask_shape[0] * 8, y8_mask_shape[1] * 8, y8_mask_shape[2] * 8)
        )

        x8 = self.path_x8(x8_f)
        x4_from_x8 = self.x8_up_x4(x8)

        x4 = self.path_x4(x4_f + x4_from_x8)
        x2_from_x4 = self.x4_up_x2(x4)

        x2_f = torch.cat([x2_f, x2_inp], dim=1)
        x2 = self.path_x2(x2_f + x2_from_x4)
        x1_from_x2 = self.x2_up_x1(x2)

        x1 = self.path_x1(torch.cat([x, x1_from_x2], dim=1))
        y1 = self.output_x1(x1)

        return y1, y1_mask
