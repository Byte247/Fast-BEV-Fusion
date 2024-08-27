import os
from mmcv.cnn import ConvModule
from torch import nn
import torch.utils.checkpoint as cp

from mmdet.models import NECKS
from mmcv.runner import auto_fp16
import ipdb


@NECKS.register_module()
class M2BevNeckTransOnly(nn.Module):
    """Neck for M2BEV.
    """

    def __init__(self,
                 is_transpose=False,
                 with_cp=False):
        super().__init__()

        self.is_transpose = is_transpose
        self.with_cp = with_cp
        
       
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """

        if bool(os.getenv("DEPLOY", False)):
            N, X, Y, Z, C = x.shape
            x = x.reshape(N, X, Y, Z*C).permute(0, 3, 1, 2)
        else:
            # N, C*T, X, Y, Z -> N, X, Y, Z, C -> N, X, Y, Z*C*T -> N, Z*C*T, X, Y
            N, C, X, Y, Z = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(N, X, Y, Z*C).permute(0, 3, 1, 2)

        if self.is_transpose:
            # Anchor3DHead axis order is (y, x).
            return [x.transpose(-1, -2)]
        else:
            return [x]
