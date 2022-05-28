"""
Fixation prediction model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        """
        Args:
            features (list of torch.Tensor): list of feature maps from the encoder
        """

        # features are expected to be:
        # layer1: [N, 128, 56, 56]
        # layer2: [N, 256, 28, 28]
        # layer3: [N, 512, 14, 14]
        # layer4: [N, 1024, 7,  7]
        pass


class FixNet(nn.Module):
    """
    UNet-like model for fixation prediction.
    """

    def __init__(self):
        super().__init__()

        # encoder is a pretrained resnext50
        backbone = models.resnext50_32x4d(pretrained=True)
        backbone_nodes = {
            "layer1.2.relu": "layer1",
            "layer2.3.relu": "layer2",
            "layer3.5.relu": "layer3",
            "layer4.2.relu": "layer4",
        }
        self.encoder = create_feature_extractor(backbone, backbone_nodes)
        self.decoder = Decoder()

    def forward(self, x):
        feats = self.encoder(x)
        for k, v in feats.items():
            print(k, v.shape)
        x = self.decoder(feats)
        return x
