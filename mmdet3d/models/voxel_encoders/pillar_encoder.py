# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from torch import nn
import numpy as np

import torch_scatter
from functools import reduce

from ..builder import VOXEL_ENCODERS
from .utils import PFNLayer, get_paddings_indicator


@VOXEL_ENCODERS.register_module()
class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True,
                 freeze_layers = False):
        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

        self.freeze_layers = freeze_layers
        if self.freeze_layers:
            for param in self.parameters():
                param.requires_grad = False
    

        

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :2])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
            else:
                f_center = features[:, :, :2]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze()
    



class PillarNet(nn.Module):
    """
    PillarNet.
    The network performs dynamic pillar scatter that convert point cloud into pillar representation
    and extract pillar features

    Reference:
    PointPillars: Fast Encoders for Object Detection from Point Clouds (https://arxiv.org/abs/1812.05784)
    End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds (https://arxiv.org/abs/1910.06528)

    Args:
        num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
    """

    def __init__(self,
                 num_input_features,
                 voxel_size,
                 pc_range):
        super().__init__()
        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)

    def forward(self, points):
        """
        Args:
            points: torch.Tensor of size (N, d), format: batch_id, x, y, z, feat1, ...
        """
        device = points.device
        dtype = points.dtype

        # discard out of range points
        grid_size = (self.pc_range[3:] - self.pc_range[:3]
                     )/self.voxel_size  # x,  y, z
        grid_size = np.round(grid_size, 0, grid_size).astype(np.int64)

        voxel_size = torch.from_numpy(
            self.voxel_size).type_as(points).to(device)
        pc_range = torch.from_numpy(self.pc_range).type_as(points).to(device)

        points_coords = (
            points[:, 1:4] - pc_range[:3].view(-1, 3)) / voxel_size.view(-1, 3)   # x, y, z

        mask = reduce(torch.logical_and, (points_coords[:, 0] >= 0,
                                          points_coords[:, 0] < grid_size[0],
                                          points_coords[:, 1] >= 0,
                                          points_coords[:, 1] < grid_size[1]))

        points = points[mask]
        points_coords = points_coords[mask]

        points_coords = points_coords.long()
        batch_idx = points[:, 0:1].long()

        points_index = torch.cat((batch_idx, points_coords[:, :2]), dim=1)
        unq, unq_inv = torch.unique(points_index, return_inverse=True, dim=0)
        unq = unq.int()

        points_mean_scatter = torch_scatter.scatter_mean(
            points[:, 1:4], unq_inv, dim=0)

        f_cluster = points[:, 1:4] - points_mean_scatter[unq_inv]

        # Find distance of x, y, and z from pillar center
        f_center = points[:, 1:3] - (points_coords[:, :2].to(dtype) * voxel_size[:2].unsqueeze(0) +
                                     voxel_size[:2].unsqueeze(0) / 2 + pc_range[:2].unsqueeze(0))

        # Combine together feature decorations
        features = torch.cat([points[:, 1:], f_cluster, f_center], dim=-1)

        return features, unq[:, [0, 2, 1]], unq_inv, grid_size[[1, 0]]
    
@VOXEL_ENCODERS.register_module()
class PillarNextPillarFeatureNet(nn.Module):
    def __init__(
        self,
        num_input_features,
        num_filters,
        voxel_size,
        pc_range,
        norm_cfg=None,
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        assert len(num_filters) > 0
        num_input_features += 5

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.feature_output_dim = num_filters[-1]

        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)

        self.voxelization = PillarNet(num_input_features, voxel_size, pc_range)

    def forward(self, points):
        features, coords, unq_inv, grid_size = self.voxelization(points)
        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)  # num_points, dim_feat

        feat_max = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]

        return feat_max, coords, grid_size



