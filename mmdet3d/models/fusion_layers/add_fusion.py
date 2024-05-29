import torch
from torch import nn

from ..builder import FUSION_LAYERS

@FUSION_LAYERS.register_module()
class AddFusion(nn.Module):
    """
    Simple and direct fusion of lidar and camera features. This module is intended as a naive baseline approach
    """
    def __init__(self,out_channels = 256):
        super(AddFusion, self).__init__()

        self.reduce_lidar_channels = nn.Conv2d(384, 256, kernel_size=1, stride=1)
        self.reduce_lidar_channels_norm = nn.BatchNorm2d(256)
        self.reduce_lidar_channels_act = nn.LeakyReLU()

        self.out_conv = nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1)
        self.out_conv_norm = nn.BatchNorm2d(out_channels)
        self.out_conv_act = nn.LeakyReLU()

        

    def forward(self, lidar_features, camera_features):

        lidar_features = self.reduce_lidar_channels_act(self.reduce_lidar_channels_norm(self.reduce_lidar_channels(lidar_features)))
        add_features = torch.add(lidar_features, camera_features)

        out = self.out_conv_act(self.out_conv_norm(self.out_conv(add_features)))

        return out


