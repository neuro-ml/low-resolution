import torch.nn as nn


class PreActivation3dNoBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv(self.activation(x))


class PostActivation3dNoBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


class ResBlock3dNoBN(nn.Module):
    def __init__(self, n_channels, kernel_size=3, padding=1, bias=True):
        super().__init__()

        self.path = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_path = self.path(x)
        return x + x_path
