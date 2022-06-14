"""
Fixation prediction model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma**2))
    return gauss / gauss.sum()


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck_up = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.decoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(n_ch * 4, n_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(n_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(n_ch, n_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(n_ch),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(n_ch, n_ch, 2, stride=2, bias=False),
                    nn.BatchNorm2d(n_ch),
                    nn.ReLU(inplace=True),
                )
                for n_ch in [256, 128, 64, 32]
            ]
        )
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 1, bias=True),
        )

    def forward(self, features):
        """
        Args:
            features (list of torch.Tensor): list of feature maps from the encoder
        """

        # features are expected to be:
        # layer0: [N, 64, 112, 112]
        # layer1: [N, 128, 56, 56]
        # layer2: [N, 256, 28, 28]
        # layer3: [N, 512, 14, 14]
        # layer4: [N, 1024, 7, 7]

        feats = list(features.values())
        x = self.bottleneck_up(feats[-1])
        for block, feat in zip(self.decoder_blocks, feats[-2::-1]):
            x = block(torch.cat([x, feat], dim=1))
        x = self.out(x)
        return x


class FixNet(nn.Module):
    """
    UNet-like model for fixation prediction.
    """

    def __init__(self, center_bias_path=None, freeze_encoder=False):
        super().__init__()

        # encoder is a pretrained resnext50
        backbone = models.resnext50_32x4d(pretrained=True)
        feature_nodes = {
            "relu": "layer0",
            "layer1.2.relu": "layer1",
            "layer2.3.relu": "layer2",
            "layer3.5.relu": "layer3",
            "layer4.2.relu": "layer4",
        }
        self.encoder = create_feature_extractor(backbone, feature_nodes)
        self.decoder = Decoder()

        # non-trainable gaussian smoothing kernel
        g = gaussian(25, 11.2)
        gaussian_kernel = torch.matmul(g.unsqueeze(-1), g.unsqueeze(-1).t())
        gaussian_kernel = gaussian_kernel.view(1, 1, 25, 25)
        self.smoothing_kernel = nn.Parameter(gaussian_kernel, requires_grad=False)

        # center bias
        if center_bias_path:
            center_bias_arr = np.load(center_bias_path)
            center_bias_log = np.log(center_bias_arr / center_bias_arr.max())
            self.center_bias = nn.Parameter(
                torch.from_numpy(center_bias_log), requires_grad=False
            )
        else:
            self.center_bias = nn.Parameter(torch.zeros(1, 224, 224), requires_grad=False)

        if freeze_encoder:
            self.set_encoder_trainable(False)

    def forward(self, x):
        feats = self.encoder(x)
        x = self.decoder(feats)

        # apply gaussian smoothing
        x = F.conv2d(x, self.smoothing_kernel, padding=12)

        # add center bias
        if self.center_bias is not None:
            x = x + self.center_bias

        return x

    def set_encoder_trainable(self, trainable: bool):
        for param in self.encoder.parameters():
            param.requires_grad = trainable
