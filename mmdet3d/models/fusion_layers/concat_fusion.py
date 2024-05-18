import torch
from torch import nn

from ..builder import FUSION_LAYERS

@FUSION_LAYERS.register_module()
class ConcatFusion(nn.Module):
    """
    Simple and direct fusion of lidar and camera features. This module is intended as a naive baseline approach
    """
    def __init__(self,out_channels = 256):
        super(ConcatFusion, self).__init__()

        self.downsample = nn.Conv2d(384 + 256, 512, kernel_size=3, stride=2, padding=1)
        self.downsample_act = nn.LeakyReLU()
        self.downsample_norm = nn.BatchNorm2d(512)

        self.conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv_norm = nn.BatchNorm2d(512)
        self.conv_act = nn.LeakyReLU()

        self.upsample = nn.ConvTranspose2d(512, out_channels, kernel_size=2, stride=2)
        self.upsample_norm = nn.BatchNorm2d(out_channels)
        self.upsample_act = nn.LeakyReLU()

    def forward(self, lidar_features, camera_features):

        
        concat_features = torch.cat([lidar_features, camera_features],dim=1)

        downsample_features = self.downsample_act(self.downsample_norm(self.downsample(concat_features)))
        middle_features = self.conv_act(self.conv_norm(self.conv(downsample_features)))
        upsample_features = self.upsample_act(self.upsample_norm(self.upsample(middle_features)))


        return upsample_features


