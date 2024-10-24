# -*- coding: utf-8 -*-
# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

voxel_size = [0.2, 0.2, 8]
out_size_factor = 4

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

score_threshold = 0.0

model = dict(
    type='FastBEVFusionTransfusionheadPillar',
    freeze_2D_backbone=True,
    freeze_2D_neck=True,
    freeze_neck_fuse=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=256, out_channels=64),
    neck_3d=dict(
        type='M2BevNeckTransOnly',
        is_transpose=False),

    #Point Modules:
    pts_voxel_layer=dict(
        max_num_points=20, voxel_size=voxel_size, max_voxels=(30000, 60000), point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64,64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        legacy=False,
        freeze_layers=True),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(512, 512)),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        conv_cfg=dict(type='Conv2d', bias=False),
        freeze_layers=True),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True,
        freeze_layers=True),

    #Fusion layer
    fusion_module = dict(type='MultiHeadCrossAttentionNoNeckSmall',
                         embed_dim = 512,
                         num_heads=1,
                         dropout = 0.1,
                         in_lidar_channels=384,
                         in_cam_channels=384,
                         output_dim = 384,
                         norm_cfg=dict(type='SyncBN', requires_grad=True),
                         one_d_norm = dict(type='SyncBN', requires_grad=True),
                         ),
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=384,
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        norm_cfg = dict(type='SyncBN', requires_grad=True),
        two_d_norm_cfg=dict(type='SyncBN', requires_grad=True),
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=[0.2, 0.2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=score_threshold,
            code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        freeze_layers=True
    ),
    
    bbox_head_2d=dict(
        type='FCOSHead',
        num_classes=10,
        in_channels=64,
        stacked_convs=2,
        feat_channels=32,
        strides=[4, 8, 16, 32],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 1e8)),
        norm_on_bbox = True,
        centerness_on_reg = True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    
    camera_n_voxels=(256, 256, 6), 
    camera_voxel_size=[0.4, 0.4, 1],

    # training and testing settings for 2d
    train_cfg_2d=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg_2d=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100),

    # model training and testing settings for the head
    train_cfg=dict(
            grid_size=[512, 512, 8],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            voxel_size=[0.2, 0.2],
            out_size_factor=out_size_factor,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range = point_cloud_range),
     test_cfg=dict(
            grid_size=[512, 512, 1],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=score_threshold,
            pc_range=point_cloud_range[:2],
            out_size_factor=out_size_factor,
            voxel_size=[0.2, 0.2],
            nms_type='rotate', #nms_type='circle',
            pre_maxsize=1000,
            post_maxsize=83,
            nms_thr=0.2)
            
)


dataset_type = 'NuScenesMultiView_Map_MultiModalDataset'
data_root = 'data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data_config = {
    'src_size': (900, 1600),
    'input_size': (900, 1600),
    # train-aug
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    # test-aug
    'test_input_size': (900, 1600),
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    # top, right, bottom, left
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}


train_pipeline = [
    dict(type='LoadAnnotations3D',
         with_bbox=True,
         with_label=True,
         with_bev_seg=False),
     dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
       type='GlobalRotScaleTrans',
       rot_range=[-0.3925 * 2, 0.3925 * 2],
       scale_ratio_range=[0.9, 1.1],
       translation_std=[0.5, 0.5, 0.5],
       update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        flip_2d=False,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='MultiViewPipeline',
        n_images=6,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
            dict(type='Pad', size_divisor=32)]),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes', 'gt_labels', 
                                 'gt_bboxes_3d', 'gt_labels_3d',
                                  'points'])]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='MultiViewPipeline',
        n_images=6,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
            dict(type='Pad', size_divisor=32)]),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img','points'])]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            with_box2d=True,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        with_box2d=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

optimizer = dict(type='AdamW', lr=1e-4,
                  weight_decay=0.01)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
     policy='cyclic',
     target_ratio=(10, 1e-4),
     cyclic_times=1,
     step_ratio_up=0.4,
 )
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)


# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)

#total_epochs = 20
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=250,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'
# load_from = None
load_additional_from = None
resume_from = None
load_from = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
workflow = [('train', 1)]



