# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt
from .multi_backbone import MultiBackbone
from .nostem_regnet import NoStemRegNet
from .pointnet2_sa_msg import PointNet2SAMSG
from .pointnet2_sa_ssg import PointNet2SASSG
from .second import SECOND
from .vovnet import VoVNet
from .swin_transformer import SwinTransformer
from .point_resnet import PointResNet34V2
from .dsvt import DSVT
from .base_bev_res_backbone import BaseBEVResBackbone
# from .dla import *


__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
    'SECOND', 'PointNet2SASSG', 'PointNet2SAMSG', 'MultiBackbone', 'PointResNet34V2',"DSVT","BaseBEVResBackbone"
]
