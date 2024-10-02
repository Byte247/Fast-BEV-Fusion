_base_ = [
    '../../_base_/datasets/nus-3d.py'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

out_size_factor = 2

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

voxel_size = [0.2, 0.2, 8]
model = dict(
    type='TransFusionHeadDSVT',
    #Point Modules:
    pts_voxel_layer=dict(
        max_num_points=20, voxel_size=voxel_size, max_voxels=(30000, 60000), point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[128,128],
        with_distance=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),
   
    dsvt_backbone=dict(
        type='DSVT',
        model_cfg=dict(
            INPUT_LAYER=dict(
                sparse_shape= [512, 512, 1],
                downsample_stride= [],
                d_model= [128],
                set_info= [[90, 4]],
                window_shape= [[30, 30, 1]],
                hybrid_factor= [1, 1, 1], # x, y, z,
                shifts_list= [[[0, 0, 0], [15, 15, 0]]],
                normalize_pos= False),
            block_name= ['DSVTBlock'],
            set_info= [[90, 4]],
            d_model= [128],
            nhead= [8],
            dim_feedforward= [256],
            dropout= 0.0,
            activation= "gelu",
            output_shape= [512, 512],
            conv_out_channel= 128)),

    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=128, output_shape=(512, 512)),

    pts_backbone=dict(
        type='BaseBEVResBackbone',
        input_channels = 128,
        LAYER_NUMS=[ 1, 2, 2 ],
        LAYER_STRIDES=[ 1, 2, 2 ],
        NUM_FILTERS= [ 128, 128, 256 ],
        UPSAMPLE_STRIDES= [ 0.5, 1, 2 ],
        NUM_UPSAMPLE_FILTERS= [ 128, 128, 128 ]),


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
        dropout=0,
        bn_momentum=0.1,
        activation='relu',
        norm_cfg=dict(type='BN1d'),
        two_d_norm_cfg=dict(type='BN'),
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    # model training and testing settings
    # model training and testing settings for the head
    train_cfg=dict(
            pts=dict(
            grid_size=[512, 512, 8],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range = point_cloud_range)),
     test_cfg=dict(
         pts=dict(
            grid_size=[512, 512, 1],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.0,
            pc_range=point_cloud_range[:2],
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            nms_type=None,
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2))
    )


dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

db_sampler = dict(
   data_root=data_root,
   info_path=data_root + 'nuscenes_dbinfos_train.pkl',
   rate=1.0,
   prepare=dict(
       filter_by_difficulty=[-1],
       filter_by_min_points=dict(
           car=5,
           truck=5,
           bus=5,
           trailer=5,
           construction_vehicle=5,
           traffic_cone=5,
           barrier=5,
           motorcycle=5,
           bicycle=5,
           pedestrian=5)),
   classes=class_names,
   sample_groups=dict(
       car=2,
       truck=3,
       construction_vehicle=7,
       bus=4,
       trailer=6,
       barrier=2,
       motorcycle=6,
       bicycle=6,
       pedestrian=2,
       traffic_cone=2),
   points_loader=dict(
       type='LoadPointsFromFile',
       coord_type='LIDAR',
       load_dim=5,
       use_dim=[0, 1, 2, 3, 4],
       file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    #dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='RandomFlip3D',
        flip_2d=False,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(512, 512),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
         type='CBGSDataset',
         dataset=dict(
             type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_train.pkl',
             pipeline=train_pipeline,
             classes=class_names,
             test_mode=False,
             use_valid_flag=True,
             # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
             # and box_type_3d='Depth' in sunrgbd and scannet dataset.
             box_type_3d='LiDAR')),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))


input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

optimizer = dict(type='AdamW', lr=0.00001,
                 weight_decay=0.05)

# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.2)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.2)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)



checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=2000,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
#load_from = "/media/tom/Volume/master_thesis/Fast-BEV-Fusion/workdirs/att_v2/fast_bev_fusion_centerhead_sub2d_att_v2/epoch_1.pth"
load_from = None
resume_from = None
workflow = [('train', 1)]

evaluation = dict(interval=1, pipeline=eval_pipeline)
