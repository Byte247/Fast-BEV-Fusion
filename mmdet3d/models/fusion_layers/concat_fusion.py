import torch
from torch import nn

from ..builder import FUSION_LAYERS

@FUSION_LAYERS.register_module()
class ConcatFusion(nn.Module):
    """
    Simple and direct fusion of lidar and camera features. This module is intended as a naive baseline approach
    """
    def __init__(self):
        super(ConcatFusion, self).__init__()

        

    def forward(self, lidar_features, camera_features):

        

        print(f"lidar_features: {lidar_features.shape}")
        print(f"camera_features: {camera_features.shape}")

        return 0


