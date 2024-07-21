# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmdet.models import DETECTORS
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import get_dist_info, auto_fp16
from mmdet3d.ops import Voxelization
from .. import builder
from mmcv.runner import force_fp32
import torch.nn.functional as F

import copy
import ipdb  # noqa


@DETECTORS.register_module()
class FastBEVFusionNoNeck(BaseDetector):
    def __init__(
        self,
        backbone,
        neck,
        neck_3d,
        bbox_head,
        camera_n_voxels,
        camera_voxel_size,
        pts_voxel_layer,
        seg_head=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_backbone=None,
        pts_neck=None,
        fusion_module=None,
        bbox_head_2d=None,
        train_cfg=None,
        test_cfg=None,
        train_cfg_2d=None,
        test_cfg_2d=None,
        pretrained=None,
        init_cfg=None,
        extrinsic_noise=0,
        with_cp=False,
        style="v1",
    ):
        super().__init__(init_cfg=init_cfg)

        #Pointa
        self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        self.pts_middle_encoder= builder.build_middle_encoder(
                pts_middle_encoder)
        self.pts_backbone = builder.build_backbone(pts_backbone)
        self.pts_neck = builder.build_neck(pts_neck)

        #Fusion

        self.fusion_module = builder.build_fusion_layer(fusion_module)

        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        
    
        self.neck_3d = builder.build_neck(neck_3d)

        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = builder.build_head(bbox_head)
            self.bbox_head.voxel_size = train_cfg.voxel_size
        else:
            self.bbox_head = None

        if seg_head is not None:
            self.seg_head = build_seg_head(seg_head)
        else:
            self.seg_head = None

        if bbox_head_2d is not None:
            bbox_head_2d.update(train_cfg=train_cfg_2d)
            bbox_head_2d.update(test_cfg=test_cfg_2d)
            self.bbox_head_2d = builder.build_head(bbox_head_2d)
        else:
            self.bbox_head_2d = None

        self.camera_n_voxels = camera_n_voxels
        self.camera_voxel_size = camera_voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise
        if self.extrinsic_noise > 0:
            for i in range(5):
                print("### extrnsic noise: {} ###".format(self.extrinsic_noise))

        # checkpoint
        self.with_cp = with_cp
        # style
        self.style = style
        assert self.style in ["v1", "v2", "v3"], self.style

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride

        #print(f"intrinsic after stride: {intrinsic}")
        extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
        #print(f"extrinsics: {extrinsics}")
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    def extract_feat(self, img, img_metas, mode):
        batch_size = img.shape[0]

        img = img.reshape(
            [-1] + list(img.shape)[2:]
        )  # [1, 6, 3, 928, 1600] -> [6, 3, 928, 1600]
        
        x = self.backbone(
            img
        )  # [6, 256, 232, 400]; [6, 512, 116, 200]; [6, 1024, 58, 100]; [6, 2048, 29, 50]

        

        # use for vovnet
        if isinstance(x, dict):
            tmp = []
            for k in x.keys():
                tmp.append(x[k])
            x = tmp

        # fuse features
        def _inner_forward(x):
            out = self.neck(x)
            return out  # [6, 64, 232, 400]; [6, 64, 116, 200]; [6, 64, 58, 100]; [6, 64, 29, 50])

        if self.with_cp and x.requires_grad:
            c1, c2, c3, c4 = cp.checkpoint(_inner_forward, x)
        else:
            c1, c2, c3, c4 = _inner_forward(x)

        features_2d = None
        if self.bbox_head_2d:
            features_2d = [c1, c2, c3, c4]

        c2 = resize(
            c2, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c3 = resize(
            c3, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c4 = resize(
            c4, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]

        x = torch.cat([c1, c2, c3, c4], dim=1)


        x = x.reshape([batch_size, -1] + list(x.shape[1:]))  # [1, 6, 64, 232, 400]

        stride = img.shape[-1] // x.shape[-1]  # 4.0
        assert stride == 4
        stride = int(stride)

        # reconstruct 3d voxels
        volumes, valids = [], []
        for feature, img_meta in zip(x, img_metas):

            # feature: [6, 64, 232, 400]
            if isinstance(img_meta["img_shape"], list):
                img_meta["img_shape"] = img_meta["img_shape"][0]
            projection = self._compute_projection(
                img_meta, stride, noise=self.extrinsic_noise
            ).to(
                x.device
            )  # [6, 3, 4]

            points = get_points(  # [3, 200, 200, 12]
                n_voxels=torch.tensor(self.camera_n_voxels),
                voxel_size=torch.tensor(self.camera_voxel_size),
                origin=torch.tensor(img_meta["lidar2img"]["origin"]),
            ).to(x.device)

            height = img_meta["img_shape"][0] // stride
            width = img_meta["img_shape"][1] // stride

            volume = backproject_inplace(feature[:, :, :height, :width], points, projection)
            volumes.append(volume)

        x = torch.stack(volumes)  # [1, 64, 200, 200, 12]
        
        def _inner_forward(x):
            out = self.neck_3d(x)  # [[1, 256, 100, 100]]
            return out

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x, None, features_2d
    
    def extract_pts_feat(self, pts):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
 
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        
        x = self.pts_neck(x)

        return x
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch



    @auto_fp16(apply_to=('img', 'points'))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            if kwargs["export_2d"]:
                return self.onnx_export_2d(img, img_metas)
            elif kwargs["export_3d"]:
                return self.onnx_export_3d(img, img_metas)
            else:
                raise NotImplementedError

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(
        self, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bev_seg=None, points=None, **kwargs
    ):  
        
        
        lidar_features = self.extract_pts_feat(points)

        camera_features, valids, features_2d = self.extract_feat(img, img_metas, "train")


        """
        feature_bev: [(1, 256, 100, 100)]
        valids: (1, 1, 200, 200, 12)
        features_2d: [[6, 64, 232, 400], [6, 64, 116, 200], [6, 64, 58, 100], [6, 64, 29, 50]]
        """

        
        #fuse lidar BEV and camera BEV features
        feature_bev = self.fusion_module(lidar_features[0], camera_features[0]) # this framework requires features inside lists for some reason. 
        feature_bev =[feature_bev]

        assert self.bbox_head is not None or self.seg_head is not None

        losses = dict()
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)

            loss_inputs = [gt_bboxes_3d, gt_labels_3d, x]
            loss_det = self.bbox_head.loss(*loss_inputs)
            losses.update(loss_det)
            

        if self.seg_head is not None:
            assert len(gt_bev_seg) == 1
            x_bev = self.seg_head(feature_bev)
            gt_bev = gt_bev_seg[0][None, ...].long()
            loss_seg = self.seg_head.losses(x_bev, gt_bev)
            losses.update(loss_seg)

        if self.bbox_head_2d is not None:
            
            batch_size = (feature_bev[0].shape)[0]
            overall_2d_loss = dict()

            for batch_id in range(batch_size):


                start_idx = batch_id * 6
                end_idx = (batch_id + 1) * 6

                # Extract the relevant slices for the current batch index
                sliced_2d_features = [tensor[start_idx:end_idx] for tensor in features_2d]

                gt_bboxes = kwargs["gt_bboxes"][batch_id]
                gt_labels = kwargs["gt_labels"][batch_id]

                # hack a img_metas_2d
                img_metas_2d = []
                img_info = img_metas[batch_id]["img_info"]
                
                for idx, info in enumerate(img_info):
                    tmp_dict = dict(
                        filename=info["filename"],
                        ori_filename=info["filename"].split("/")[-1],
                        ori_shape=img_metas[batch_id]["ori_shape"],
                        img_shape=img_metas[batch_id]["img_shape"],
                        pad_shape=img_metas[batch_id]["pad_shape"],
                        scale_factor=img_metas[batch_id]["scale_factor"],
                        flip=False,
                        flip_direction=None,
                    )
                    img_metas_2d.append(tmp_dict)

                rank, world_size = get_dist_info()


                loss_2d = self.bbox_head_2d.forward_train(
                    sliced_2d_features, img_metas_2d, gt_bboxes, gt_labels
                )
                
                # Check for NaN in loss_2d and handle it
                for key, value in loss_2d.items():
                    if torch.isnan(value).any():
                        print(f"NaN detected in {key} for batch_id {batch_id}, replacing with zero.")
                        loss_2d[key] = torch.zeros_like(value)

                if batch_id == 0:
                    overall_2d_loss.update(loss_2d)
                else:
                    for key in overall_2d_loss:
                        overall_2d_loss[key] += loss_2d[key]

            # Normalize the loss by batch size outside the loop
            for key in overall_2d_loss:
                if torch.isnan(overall_2d_loss[key]).any():
                    print(f"NaN detected in overall_2d_loss before normalization in {key}, replacing with zero.")
                    overall_2d_loss[key] = torch.zeros_like(overall_2d_loss[key])
                overall_2d_loss[key] /= batch_size

            # Check for NaN after normalization and handle it
            for key in overall_2d_loss:
                if torch.isnan(overall_2d_loss[key]).any():
                    print(f"NaN detected in overall_2d_loss after normalization in {key}, replacing with zero.")
                    overall_2d_loss[key] = torch.zeros_like(overall_2d_loss[key])

            # Update losses
            losses.update(overall_2d_loss)
            
        return losses

    def forward_test(self, img, img_metas, points,**kwargs): 
        if not self.test_cfg.get('use_tta', False):
            return self.simple_test(img, img_metas, points)
        return self.aug_test(img, img_metas)

    def onnx_export_2d(self, img, img_metas):
        """
        input: 6, 3, 544, 960
        output: 6, 64, 136, 240
        """
        x = self.backbone(img)
        c1, c2, c3, c4 = self.neck(x)
        c2 = resize(
            c2, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c3 = resize(
            c3, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c4 = resize(
            c4, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.neck_fuse(x)

        if bool(os.getenv("DEPLOY", False)):
            x = x.permute(0, 2, 3, 1)
            return x

        return x

    def onnx_export_3d(self, x, _):
        # x: [6, 200, 100, 3, 256]
        # if bool(os.getenv("DEPLOY_DEBUG", False)):
        #     x = x.sum(dim=0, keepdim=True)
        #     return [x]
        if self.style == "v1":
            x = x.sum(dim=0, keepdim=True)  # [1, 200, 100, 3, 256]
            x = self.neck_3d(x)  # [[1, 256, 100, 50], ]
        elif self.style == "v2":
            x = self.neck_3d(x)  # [6, 256, 100, 50]
            x = [x[0].sum(dim=0, keepdim=True)]  # [1, 256, 100, 50]
        elif self.style == "v3":
            x = self.neck_3d(x)  # [1, 256, 100, 50]
        else:
            raise NotImplementedError

        if self.bbox_head is not None:
            cls_score, bbox_pred, dir_cls_preds = self.bbox_head(x)
            cls_score = [item.sigmoid() for item in cls_score]

        if os.getenv("DEPLOY", False):
            x = [cls_score, bbox_pred, dir_cls_preds]
            return x

        return x
    

    def simple_test(self, img, img_metas, points):
        bbox_results = []
        feature_bev, _, features_2d = self.extract_feat(img, img_metas, "test")

        lidar_features = self.extract_pts_feat(points)

        #fuse lidar BEV and camera BEV features
        feature_bev = self.fusion_module(lidar_features[0], feature_bev[0])
        feature_bev =[feature_bev]


        if self.bbox_head is not None:
            outs = self.bbox_head(feature_bev)
            bbox_list = self.bbox_head.get_bboxes(outs, img_metas, rescale=True)
                                    
            bbox_results = [bbox3d2result(bboxes, scores, labels)for bboxes, scores, labels in bbox_list]
            
            
        else:
            bbox_results = [dict()]

        # BEV semantic seg
        if self.seg_head is not None:
            x_bev = self.seg_head(feature_bev)
            bbox_results[0]['bev_seg'] = x_bev

        return bbox_results

    def aug_test(self, imgs, img_metas):
        img_shape_copy = copy.deepcopy(img_metas[0]['img_shape'])
        extrinsic_copy = copy.deepcopy(img_metas[0]['lidar2img']['extrinsic'])

        x_list = []
        img_metas_list = []
        for tta_id in range(2):

            img_metas[0]['img_shape'] = img_shape_copy[24*tta_id:24*(tta_id+1)]
            img_metas[0]['lidar2img']['extrinsic'] = extrinsic_copy[24*tta_id:24*(tta_id+1)]
            img_metas_list.append(img_metas)

            feature_bev, _, _ = self.extract_feat(imgs[:, 24*tta_id:24*(tta_id+1)], img_metas, "test")
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results

    def show_results(self, *args, **kwargs):
        pass


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_inplace(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume