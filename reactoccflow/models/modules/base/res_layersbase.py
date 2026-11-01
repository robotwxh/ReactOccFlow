import torch.nn as nn
from .others import Normalize, nonlinearity

class Residual_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.norm = Normalize(out_channel)

    def forward(self, latent_features):
        conv1 = self.conv1(latent_features)
        # conv1 = rearrange(conv1, 'b c h w -> b h w c')
        input_dtype = conv1.dtype
        conv1 = self.norm(conv1).to(input_dtype)
        conv1 = nonlinearity(conv1)
        conv2 = self.conv2(conv1)
        if conv2.shape[1] != latent_features.shape[1]:
            return conv2
        return conv2 + latent_features


class ResidualConv3D(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=None, nonlinearity=nn.SiLU()):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channel)
        self.nonlinearity = nonlinearity

    def forward(self, latent_features):
        x = self.conv1(latent_features)
        x = self.norm(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)

        if x.shape[1] != latent_features.shape[1]:
            return x  # No residual if channels mismatch
        return x + latent_features
