# Copyright (c) OpenMMLab. All rights reserved.
import copy
import mmcv
import numpy as np
import os
import cv2

from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.pipelines import data_augment_utils
from mmdet.datasets import PIPELINES
from ..builder import OBJECTSAMPLERS


class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str | None): The category of samples. Default: None.
        epoch (int | None): Sampling epoch. Default: None.
        shuffle (bool): Whether to shuffle indices. Default: False.
        drop_reminder (bool): Drop reminder. Default: False.
    """

    def __init__(self,
                 sampled_list,
                 name=None,
                 epoch=None,
                 shuffle=True,
                 drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


@OBJECTSAMPLERS.register_module()
class DataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3])):
        super().__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES)

        db_infos = mmcv.load(info_path)

        # filter database infos
        from mmdet3d.utils import get_root_logger
        logger = get_root_logger()
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')

        self.db_infos = db_infos

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self, gt_bboxes, gt_labels, img=None):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]

            # num_sampled = len(sampled)
            s_points_list = []
            count = 0
            for info in sampled:
                file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                results = dict(pts_filename=file_path)
                s_points = self.points_loader(results)['points']
                s_points.translate(info['box3d_lidar'][:3])

                count += 1

                s_points_list.append(s_points)

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.longlong)
            ret = {
                'gt_labels_3d':
                gt_labels,
                'gt_bboxes_3d':
                sampled_gt_bboxes,
                'points':
                s_points_list[0].cat(s_points_list),
                'group_ids':
                np.arange(gt_bboxes.shape[0],
                          gt_bboxes.shape[0] + len(sampled))
            }

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

@OBJECTSAMPLERS.register_module()
class MMDataBaseSampler(DataBaseSampler):

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 check_2D_collision=False,
                 collision_thr=0,
                 collision_in_classes=False,
                 depth_consistent=False,
                 blending_type=None,
                 img_loader=dict(type='LoadImageFromFile'),
                 mask_loader=dict(
                     type='LoadImageFromFile', color_type='grayscale'),
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     load_dim=4,
                     coord_type='LIDAR',
                     use_dim=[0, 1, 2, 3])):
        super(MMDataBaseSampler, self).__init__(
            info_path=info_path,
            data_root=data_root,
            rate=rate,
            prepare=prepare,
            sample_groups=sample_groups,
            classes=classes,
            points_loader=points_loader)

        self.blending_type = blending_type
        self.depth_consistent = depth_consistent
        self.check_2D_collision = check_2D_collision
        self.collision_thr = collision_thr
        self.collision_in_classes = collision_in_classes
        self.img_loader = mmcv.build_from_cfg(img_loader, PIPELINES)
        self.mask_loader = mmcv.build_from_cfg(mask_loader, PIPELINES)

    def sample_all(self, gt_bboxes_3d, gt_names, gt_bboxes_2d=None, img=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            sampled_num = int(max_sample_num -
                              np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes_3d = []
        sampled_gt_bboxes_2d = []
        avoid_coll_boxes_3d = gt_bboxes_3d
        avoid_coll_boxes_2d = gt_bboxes_2d

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes_3d,
                                                   avoid_coll_boxes_2d)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box_3d = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                        sampled_gt_box_2d = sampled_cls[0]['box2d_camera'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box_3d = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)
                        sampled_gt_box_2d = np.stack(
                            [s['box2d_camera'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes_3d += [sampled_gt_box_3d]
                    sampled_gt_bboxes_2d += [sampled_gt_box_2d]
                    if self.collision_in_classes:
                        # TODO: check whether check collision check among
                        # classes is necessary
                        avoid_coll_boxes_3d = np.concatenate(
                            [avoid_coll_boxes_3d, sampled_gt_box_3d], axis=0)
                        avoid_coll_boxes_2d = np.concatenate(
                            [avoid_coll_boxes_2d, sampled_gt_box_2d], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes_3d = np.concatenate(sampled_gt_bboxes_3d, axis=0)
            sampled_gt_bboxes_2d = np.concatenate(sampled_gt_bboxes_2d, axis=0)

            s_points_list = []
            count = 0

            if self.depth_consistent:
                # change the paster order based on distance
                center = sampled_gt_bboxes_3d[:, 0:3]
                paste_order = np.argsort(
                    -np.power(np.sum(np.power(center, 2), axis=-1), 1 / 2),
                    axis=-1)

            for idx in range(len(sampled)):
                if self.depth_consistent:
                    inds = np.where(paste_order == idx)[0][0]
                    info = sampled[inds]
                else:
                    info = sampled[idx]

                pcd_file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                img_file_path = pcd_file_path + '.png'
                mask_file_path = pcd_file_path + '.mask.png'

                results = dict(pts_filename=pcd_file_path)
                s_points = self.points_loader(results)['points']
                # perform points sampling before pasting

                patch_results = dict(
                    img_prefix=None, img_info=dict(filename=img_file_path))
                mask_results = dict(
                    img_prefix=None, img_info=dict(filename=mask_file_path))
                s_patch = self.img_loader(patch_results)['img']
                s_mask = self.mask_loader(mask_results)['img']

                # the points of each sample already minus the object center
                # so this time it needs to add the offset back
                s_points.translate(info['gt_bboxes_3d'][:3])
                img = self.paste_obj(
                    img,
                    s_patch,
                    s_mask,
                    bbox_2d=info['box2d_camera'].astype(np.int32))

                count += 1
                s_points_list.append(s_points)

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled])
            ret = dict(
                img=img,
                gt_labels=gt_labels,
                gt_labels_3d=copy.deepcopy(gt_labels),
                gt_bboxes_3d=sampled_gt_bboxes_3d,
                gt_bboxes_2d=sampled_gt_bboxes_2d,
                points=s_points_list[0].cat(s_points_list),
                group_ids=np.arange(gt_bboxes_3d.shape[0],
                                    gt_bboxes_3d.shape[0] + len(sampled)))

        return ret

    def paste_obj(self, img, obj_img, obj_mask, bbox_2d):
        # paste the image patch back
        x1, y1, x2, y2 = bbox_2d
        # the bbox might exceed the img size because the img is different
        img_h, img_w = img.shape[:2]
        w = np.maximum(min(x2, img_w - 1) - x1 + 1, 1)
        h = np.maximum(min(y2, img_h - 1) - y1 + 1, 1)
        obj_mask = obj_mask[:h, :w]
        obj_img = obj_img[:h, :w]

        # choose a blend option
        if not self.blending_type:
            blending_op = 'none'

        else:
            blending_choice = np.random.randint(len(self.blending_type))
            blending_op = self.blending_type[blending_choice]

        if blending_op.find('poisson') != -1:
            # options: cv2.NORMAL_CLONE=1, or cv2.MONOCHROME_TRANSFER=3
            # cv2.MIXED_CLONE mixed the texture, thus is not used.
            if blending_op == 'poisson':
                mode = np.random.choice([1, 3], 1)[0]
            elif blending_op == 'poisson_normal':
                mode = cv2.NORMAL_CLONE
            elif blending_op == 'poisson_transfer':
                mode = cv2.MONOCHROME_TRANSFER
            else:
                raise NotImplementedError
            center = (int(x1 + w / 2), int(y1 + h / 2))
            img = cv2.seamlessClone(obj_img, img, obj_mask * 255, center, mode)
        else:
            if blending_op == 'gaussian':
                obj_mask = cv2.GaussianBlur(
                    obj_mask.astype(np.float32), (5, 5), 2)
            elif blending_op == 'box':
                obj_mask = cv2.blur(obj_mask.astype(np.float32), (3, 3))
            paste_mask = 1 - obj_mask
            img[y1:y1 + h,
                x1:x1 + w] = (img[y1:y1 + h, x1:x1 + w].astype(np.float32) *
                              paste_mask[..., None]).astype(np.uint8)
            img[y1:y1 + h, x1:x1 + w] += (obj_img.astype(np.float32) *
                                          obj_mask[..., None]).astype(np.uint8)

        return img

    def sample_class_v2(self, name, num, gt_bboxes_3d, gt_bboxes_2d):
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes_3d.shape[0]
        num_sampled = len(sampled)
        # avoid collision in BEV first
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes_3d[:, 0:2], gt_bboxes_3d[:, 3:5], gt_bboxes_3d[:, 6])
        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes[:, 0:2], sp_boxes[:, 3:5], sp_boxes[:, 6])
        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)

        for element in sampled:
            print(element.keys())

        # Then avoid collision in 2D space
        if self.check_2D_collision:
            sp_boxes_2d = np.stack([i['box2d_camera'] for i in sampled],
                                   axis=0)
            total_bbox_2d = np.concatenate([gt_bboxes_2d, sp_boxes_2d],
                                           axis=0)  # Nx4
            # random select a collision threshold
            if isinstance(self.collision_thr, float):
                collision_thr = self.collision_thr
            elif isinstance(self.collision_thr, list):
                collision_thr = np.random.choice(self.collision_thr)
            elif isinstance(self.collision_thr, dict):
                mode = self.collision_thr.get('mode', 'value')
                if mode == 'value':
                    collision_thr = np.random.choice(
                        self.collision_thr['thr_range'])
                elif mode == 'range':
                    collision_thr = np.random.uniform(
                        self.collision_thr['thr_range'][0],
                        self.collision_thr['thr_range'][1])

            if collision_thr == 0:
                # use similar collision test as BEV did
                # Nx4 (x1, y1, x2, y2) -> corners: Nx4x2
                # ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
                x1y1 = total_bbox_2d[:, :2]
                x2y2 = total_bbox_2d[:, 2:]
                x1y2 = np.stack([total_bbox_2d[:, 0], total_bbox_2d[:, 3]],
                                axis=-1)
                x2y1 = np.stack([total_bbox_2d[:, 2], total_bbox_2d[:, 1]],
                                axis=-1)
                total_2d = np.stack([x1y1, x2y1, x1y2, x2y2], axis=1)
                coll_mat_2d = data_augment_utils.box_collision_test(
                    total_2d, total_2d)
            else:
                # use iof rather than iou to protect the foreground
                overlaps = box_np_ops.iou_jit(total_bbox_2d, total_bbox_2d,
                                              'iof')
                coll_mat_2d = overlaps > collision_thr
            coll_mat = coll_mat + coll_mat_2d

        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])

        return valid_samples