# Based on https://github.com/nutonomy/nuscenes-devkit
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import mmcv
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample
import imageio




cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams

class Visualizer:

    def __init__(self) -> None:
        
        
        pass


    def get_sample_data(self, sample_data_token: str,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        selected_anntokens=None,
                        use_flat_vehicle_coordinates: bool = False):
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param selected_anntokens: If provided only return the selected annotation.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                            aligned to z-plane in the world.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """
        
        # Retrieve sensor & pose records
        sd_record = nusc.get('sample_data', sample_data_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        data_path = nusc.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize =(sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(nusc.get_box, selected_anntokens))
        else:
            boxes = nusc.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            
            if use_flat_vehicle_coordinates:
                
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            else:
                
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                #Move box to sensor coord system.
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

            if sensor_record['modality'] == 'camera' and not \
                    box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue

            box_list.append(box)

        return data_path, box_list, cam_intrinsic



    def get_predicted_data(self, sample_data_token: str,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        selected_anntokens=None,
                        use_flat_vehicle_coordinates: bool = False,
                        pred_anns=None
                        ):
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param selected_anntokens: If provided only return the selected annotation.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                            aligned to z-plane in the world.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """

        # Retrieve sensor & pose records
        sd_record = nusc.get('sample_data', sample_data_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        data_path = nusc.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            cam_intrinsic[:2, :] /= 2.0
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        boxes = pred_anns

        box_list = []
        for box in boxes:
            # Transform box to ego vehicle coordinate system
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            # Transform box to sensor coordinate system
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

            if sensor_record['modality'] == 'camera' and not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue

            box_list.append(box)

        return data_path, box_list, cam_intrinsic




    def lidiar_render(self, sample_token, data,out_path=None):
        bbox_gt_list = []
        bbox_pred_list = []
        anns = nusc.get('sample', sample_token)['anns']
        for ann in anns:
            content = nusc.get('sample_annotation', ann)
            try:
                bbox_gt_list.append(DetectionBox(
                    sample_token=content['sample_token'],
                    translation=tuple(content['translation']),
                    size=tuple(content['size']),
                    rotation=tuple(content['rotation']),
                    velocity=nusc.box_velocity(content['token'])[:2],
                    ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                    else tuple(content['ego_translation']),
                    num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                    detection_name=category_to_detection_name(content['category_name']),
                    detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                    attribute_name=''))
            except:
                pass

        bbox_anns = data['results'][sample_token]
        for content in bbox_anns:
            bbox_pred_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=tuple(content['velocity']),
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=content['detection_name'],
                detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=content['attribute_name']))
        gt_annotations = EvalBoxes()
        pred_annotations = EvalBoxes()
        gt_annotations.add_boxes(sample_token, bbox_gt_list)
        pred_annotations.add_boxes(sample_token, bbox_pred_list)
        print('green is ground truth')
        print('blue is the predited result')
        visualize_sample(nusc, sample_token, gt_annotations, pred_annotations, savepath=out_path+'_bev_'+sample_token)


    def get_color(self,category_name: str):
        """
        Provides the default colors based on the category names.
        This method works for the general nuScenes categories, as well as the nuScenes detection categories.
        """
        a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
        'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
        'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
        'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
        'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
        'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
        'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
        'vehicle.ego']
        class_names = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
        #print(category_name)
        if category_name == 'bicycle':
            return nusc.colormap['vehicle.bicycle']
        elif category_name == 'construction_vehicle':
            return nusc.colormap['vehicle.construction']
        elif category_name == 'traffic_cone':
            return nusc.colormap['movable_object.trafficcone']

        for key in nusc.colormap.keys():
            if category_name in key:
                return nusc.colormap[key]
        return [0, 0, 0]


    def render_sample_data(
            self,
            sample_token: str,
            with_anns: bool = True,
            box_vis_level: BoxVisibility = BoxVisibility.ANY,
            axes_limit: float = 40,
            ax=None,
            nsweeps: int = 1,
            out_path: str = None,
            underlay_map: bool = True,
            use_flat_vehicle_coordinates: bool = True,
            show_lidarseg: bool = False,
            show_lidarseg_legend: bool = False,
            filter_lidarseg_labels=None,
            lidarseg_preds_bin_path: str = None,
            verbose: bool = True,
            show_panoptic: bool = False,
            pred_data=None,
        ) -> None:
        """
        Render sample data onto axis.
        """
        self.lidiar_render(sample_token, pred_data, out_path=out_path)
        sample = nusc.get('sample', sample_token)
        cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        if ax is None:
            _, ax = plt.subplots(4, 3, figsize=(24, 18))
        j = 0
        for ind, cam in enumerate(cams):
            sample_data_token = sample['data'][cam]

            sd_record = nusc.get('sample_data', sample_data_token)
            sensor_modality = sd_record['sensor_modality']

            # Load boxes and image.
            boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                            name=record['detection_name'], token='predicted') for record in
                        pred_data['results'][sample_token] if record['detection_score'] > 0.2]
            
            
            data_path, boxes_pred, camera_intrinsic = self.get_predicted_data(sample_data_token,
                                                                            box_vis_level=box_vis_level, pred_anns=boxes, use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)
            
            
            _, boxes_gt, _ = self.get_sample_data(sample_data_token, box_vis_level=box_vis_level, use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)
            if ind == 3:
                j += 1
            ind = ind % 3
            data = Image.open(data_path)
            # Init axes.

            # Show image.
            ax[j, ind].imshow(data)
            ax[j + 2, ind].imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes_pred:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax[j, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))
                for box in boxes_gt:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax[j + 2, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax[j, ind].set_xlim(0, data.size[0])
            ax[j, ind].set_ylim(data.size[1], 0)
            ax[j + 2, ind].set_xlim(0, data.size[0])
            ax[j + 2, ind].set_ylim(data.size[1], 0)

            ax[j, ind].axis('off')
            ax[j, ind].set_title('PRED: {} {labels_type}'.format(
                sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
            ax[j, ind].set_aspect('equal')

            ax[j + 2, ind].axis('off')
            ax[j + 2, ind].set_title('GT:{} {labels_type}'.format(
                sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
            ax[j + 2, ind].set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path+'_camera_' + sample_token, bbox_inches='tight', pad_inches=0, dpi=400)
        if verbose:
            plt.show()
        plt.close()




    

if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)
    bevformer_results = mmcv.load('/home/tom/ws/Fast-BEV-Fusion/work_dirs/fast_bev_fusion_centerhead_sub2d_att_v2/results_nusc.json')
    sample_token_list = list(bevformer_results['results'].keys())

    vis = Visualizer()
    for id in range(0, 100):
       vis.render_sample_data(sample_token_list[id], pred_data=bevformer_results, out_path="/home/tom/ws/Fast-BEV-Fusion/work_dirs/fast_bev_fusion_centerhead_sub2d_att_v2/figs/", use_flat_vehicle_coordinates=False, verbose=False)

