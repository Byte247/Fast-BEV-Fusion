import numpy as np
import spconv
from spconv.pytorch.conv import SparseConv3d,SubMConv3d, SparseConv2d
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.core import ConvAlgo

from torch import nn
import torch

from mmcv.cnn import build_norm_layer

from ..builder import MIDDLE_ENCODERS,BACKBONES



def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


class SparseConvBlock(spconv.pytorch.SparseModule):
    '''
    Sparse Conv Block
    SparseConv2d for stride > 1 and subMconv2d for stride==1
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, use_subm=True, bias=False, norm_cfg=None):
        super(SparseConvBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)


        if stride == 1 and use_subm:
            self.conv = spconv.pytorch.SubMConv2d(in_channels, out_channels, kernel_size,
                                                  padding=kernel_size//2, stride=1, bias=bias)
        else:
            self.conv = spconv.pytorch.SparseConv2d(in_channels, out_channels, kernel_size,
                                                    padding=kernel_size//2, stride=stride, bias=bias)
       
        self.norm = build_norm_layer(norm_cfg, out_channels)[1],
        self.act = nn.LeakyReLU()

    def forward(self, x):
        print(f"in sparse ConvBlock {x.features.shape}")
        print(f"conv:{self.conv}")
        out = self.conv(x)
        out = out.replace_feature(self.norm(out.features))
        out = out.replace_feature(self.act(out.features))

        return out


class SparseBasicBlock(spconv.pytorch.SparseModule):
    '''
    Sparse Conv Block
    '''

    def __init__(self, channels, kernel_size, norm_cfg=None):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.block1 = SparseConvBlock(channels, channels, kernel_size, 1)
        self.conv2 = spconv.pytorch.SubMConv2d(channels, channels, kernel_size, padding=kernel_size//2,
                                               stride=1, bias=False, algo=ConvAlgo.Native, )
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        self.act2 = nn.LeakyReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.act2(out.features))

        return out


class SparseBasicBlock3D(SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock3D, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.LeakyReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(torch.add(out.features,identity.features))
        out = out.replace_feature(self.relu(out.features))

        return out


@MIDDLE_ENCODERS.register_module()
class SpMiddleResNetFHD(nn.Module):
    def __init__(
        self, in_channels=5, norm_cfg=None, sparse_shape=[41, 1024, 1024], name="SpMiddleResNetFHD", **kwargs
    ):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False
        self.sparse_shape = sparse_shape

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.conv_input = SparseSequential(
            SubMConv3d(in_channels, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.LeakyReLU(inplace=True)
        )

        self.conv1 = SparseSequential(        
            SparseBasicBlock3D(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock3D(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = SparseSequential(
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.LeakyReLU(inplace=True),
            SparseBasicBlock3D(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock3D(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = SparseSequential(
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.LeakyReLU(inplace=True),
            SparseBasicBlock3D(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock3D(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = SparseSequential(
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.LeakyReLU(inplace=True),
            SparseBasicBlock3D(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock3D(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )


        self.extra_conv = SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.LeakyReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):

        coors = coors.int()
        ret = SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)


        second_output = x_conv3.dense()
        N, C, D, H, W = second_output.shape
        second_output = second_output.view(N, C * D, H, W)

        ret = self.extra_conv(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)


        return [ret, second_output] #, multi_scale_voxel_features
    








@BACKBONES.register_module()
class SparseResNet18(nn.Module):
    def __init__(
            self,
            layer_nums,
            ds_layer_strides,
            ds_num_filters,
            num_input_features,
            kernel_size=[3, 3, 3, 3],
            out_channels=256,
            sparse_shape=[1, 512, 512],
            norm_cfg=None):

        super(SparseResNet18, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features
        self.sparse_shape = sparse_shape

        if norm_cfg is not None:
            self.norm_cfg = norm_cfg
        else:
            self.norm_cfg = None

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                kernel_size[i],
                self._layer_strides[i],
                layer_num)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        if self.norm_cfg is not None:
            mapping_norm = build_norm_layer(self.norm_cfg, out_channels)[1]
        else:
            mapping_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

        
        self.mapping = SparseSequential(
            SparseConv2d(self._num_filters[-1],
                         out_channels, 1, 1, bias=False),
            mapping_norm,
            nn.LeakyReLU(),
        )

    def _make_layer(self, inplanes, planes, kernel_size, stride, num_blocks):

        layers = []
        layers.append(SparseConvBlock(inplanes, planes,
                      kernel_size=kernel_size, stride=stride, use_subm=False, norm_cfg=self.norm_cfg))

        for j in range(num_blocks):
            layers.append(SparseBasicBlock(planes, kernel_size=kernel_size, norm_cfg=self.norm_cfg))

        return spconv.pytorch.SparseSequential(*layers)
    

    def forward(self, pillar_features, coors, input_shape):
        batch_size = len(torch.unique(coors[:, 0]))
        x = spconv.pytorch.SparseConvTensor(
            pillar_features, coors, input_shape, batch_size)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.mapping(x)
        return x.dense()