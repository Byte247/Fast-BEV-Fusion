_base_ = [
    '../../_base_/datasets/nus-3d.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


voxel_size = [0.2, 0.2, 8]
model = dict(
    type='CenterPointPretrain',
    #Point Modules:
    pts_voxel_layer=dict(
        max_num_points=20, voxel_size=[0.2, 0.2, 8], max_voxels=(30000, 60000), point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='PillarNextPillarFeatureNet',
        num_input_features=5,
        num_filters=[64,64],
        voxel_size=(0.2, 0.2, 8),
        pc_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', requires_grad=True)),
    pts_backbone=dict(
        type='SparseResNet18',
        num_input_features=64,
        layer_nums=[2,2,2,2],
        ds_layer_strides=[1,2,2,2],
        ds_num_filters=[64,128,256,256],
        out_channels=256,
        sparse_shape=[1, 512, 512],
        norm_cfg=dict(type='BN', requires_grad=True)),
    pts_neck=dict(
        type='ASPPNeck',
        in_channels=256, 
        out_channels=384,
        norm_cfg=dict(type='BN', requires_grad=True)),
    bbox_head= dict(
        type='CenterHead',
        in_channels=384,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=128,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2],
            pc_range=[-51.2, -51.2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            point_cloud_range=point_cloud_range,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=point_cloud_range[:2],
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))


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
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
       type='GlobalRotScaleTrans',
       rot_range=[-0.3925, 0.3925],
       scale_ratio_range=[0.95, 1.05],
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
        sweeps_num=9,
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
        sweeps_num=9,
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
    samples_per_gpu=4,
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

optimizer = dict(type='AdamW', lr=0.0001,
                 weight_decay=0.01)

# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.001),
    cyclic_times=1,
    step_ratio_up=0.3)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.3)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)



checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=100,
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
