from utils import augmentor_utils
import numpy as np
import json
from ops.iou3d_nms import iou3d_nms_utils
import utils.box_utils as box_utils


class PointsWithBoxes3DAugmentor(object):
    def __init__(self, category_name_gt_lut_full_path, augmentation_config):
        """
        Constructor

        :param category_name_gt_lut_full_path: full path of training ground truth (boxes) look-up table
        :param augmentation_config: configuration of augmentation
        """
        self._category_name_gt_lut_full_path = category_name_gt_lut_full_path
        self._augmentation_config = augmentation_config

        # load ground truth samples pool from training dataset
        self._train_gt_boxes_pool = json.load(open(self._category_name_gt_lut_full_path, "r"))

        self._limit_whole_scene = self._augmentation_config["LIMIT_WHOLE_SCENE"]
        self._sample_groups_config = self._augmentation_config["SAMPLING_GROUPS"]
        self._extra_width = self._augmentation_config["EXTRA_WIDTH"]

        # build samples group look-up table (LUT)
        self._sample_groups = dict()
        self._sample_category_num = dict()
        for category_name, sample_num in self._sample_groups_config:
            self._sample_category_num[category_name] = sample_num
            self._sample_groups[category_name] = {
                "sample_num": sample_num,
                "pointer": len(self._train_gt_boxes_pool[category_name]),
                "indices": np.arange(len(self._train_gt_boxes_pool[category_name]))
            }

    def sample_with_fixed_number(self, category_name, sample_group):
        """
        Sample with fixed number boxes from self._train_gt_boxes_pool

        :param category_name: string
        :param sample_group: a dictionary with "sample_num", "pointer" and "indices"
        :return:
        """
        sample_num = int(sample_group["sample_num"])
        pointer = sample_group["pointer"]
        indices = sample_group["indices"]

        # if sampled index arrives the end of gt samples list, then random shuffle the indices and reset pointer to 0
        if pointer >= len(self._train_gt_boxes_pool[category_name]):
            indices = np.random.permutation(len(self._train_gt_boxes_pool[category_name]))
            pointer = 0

        sampled_gt_list = list()
        # attention: not ensure each sampling statisfy the sample_num requirement if
        # pointer + sample_num > len(self._train_gt_boxes_pool[category_name])
        for idx in indices[pointer:pointer+sample_num]:
            sampled_gt_list.append(self._train_gt_boxes_pool[category_name][idx])

        pointer += sample_num
        sample_group["pointer"] = pointer
        sample_group["indices"] = indices

        return sampled_gt_list

    def place_sampled_objects_on_current_points(self, raw_points, raw_gt_boxes, raw_gt_names):
        """
        Place sampled ground truth objects on current lidar point cloud

        :param raw_points: lidar points, np.ndarray, shape is (N, 4), 4 indicates (x, y, z, intensity)
        :param raw_gt_boxes: original ground truth boxes, shape is (M, 7), 7 indicates (x, y, z, l, w, h, orientation)
        :param raw_gt_names: (M, ), string, np.ndarray
        :return:
        """
        total_valid_sampled_dict = list()
        existed_boxes = raw_gt_boxes

        for category_name, sample_group in self._sample_groups.items():
            if self._limit_whole_scene:
                # limit the total amount of a given category name, the original existing boxes of this category plus
                # the augmented ones should not be larger than target value self._sample_category_num[category_name]
                num_gt = np.sum(category_name == raw_gt_names)
                sample_group['sample_num'] = self._sample_category_num[category_name] - num_gt

            # if self._sample_category_num[category_name] - num_gt is larger than 0, then do boxes augmentation
            if sample_group['sample_num'] > 0:
                sampled_dict = self.sample_with_fixed_number(category_name=category_name, sample_group=sample_group)
                sampled_boxes = np.stack([x['box3d'] for x in sampled_dict], axis=0).astype(np.float32)

                # calculate the IoU between sampled boxes and original existing boxes
                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], raw_gt_boxes[:, 0:7])

                # calculate the IoU between sampled boxes
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])

                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate((raw_gt_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[raw_gt_boxes.shape[0]:, :]
        if len(total_valid_sampled_dict) > 0:
            points_aug, gt_boxes_aug, gt_names_aug = self.add_sampled_boxes_to_scene(
                raw_points=raw_points,
                raw_gt_boxes=raw_gt_boxes,
                raw_gt_names=raw_gt_names,
                sampled_gt_boxes=sampled_gt_boxes,
                total_valid_sampled_dict=total_valid_sampled_dict)
        else:
            points_aug, gt_boxes_aug, gt_names_aug = raw_points, raw_gt_boxes, raw_gt_names

        return points_aug, gt_boxes_aug, gt_names_aug

    def add_sampled_boxes_to_scene(self, raw_points, raw_gt_boxes, raw_gt_names,
                                   sampled_gt_boxes, total_valid_sampled_dict):
        """
        Append sampled boxes and their corresponding points cloud to original ground truth and points

        :param raw_points: original points, np.ndarray, shape is (N, 4)
        :param raw_gt_boxes: original ground truth bounding boxes, shape is (M, 7)
        :param raw_gt_names: (M, ), string, np.ndarray
        :param sampled_gt_boxes: sampled bounding boxes, shape is (K, 7)
        :param total_valid_sampled_dict: list of dictionary, length is K
        :return:
        """
        obj_points_list = list()

        for idx, info in enumerate(total_valid_sampled_dict):
            gt_box_npy_full_path = info["path"]
            obj_points = np.load(gt_box_npy_full_path)

            obj_points[:, :3] += info["box3d"][:3]
            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x["category"] for x in total_valid_sampled_dict])

        # enlarge the ground truth bounding boxes
        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            boxes3d=sampled_gt_boxes[:, 0:7],
            extra_width=self._extra_width)

        # remove points in sampled bounding boxes
        points = box_utils.remove_points_in_boxes3d(raw_points, large_sampled_gt_boxes)

        # concatenate points with sampled objects' points
        points_aug = np.concatenate([obj_points, points], axis=0)
        gt_boxes_aug = np.concatenate([raw_gt_boxes, sampled_gt_boxes], axis=0)
        gt_names_aug = np.concatenate([raw_gt_names, sampled_gt_names], axis=0)

        return points_aug, gt_boxes_aug, gt_names_aug

    def random_world_flip(self, points, gt_boxes, gt_names):
        """
        Flip the world (points cloud) randomly

        :param points: (N, 4), 4 indicates (x, y, z, intensity)
        :param gt_boxes: (N, 7), 7 indicates (x, y, z, l, w, h, orientation)
        :param gt_names: (N, ), string
        :return:
        """
        flip_axis_list = self._augmentation_config["RANDOM_WORLD_FLIP"]["ALONG_AXIS_LIST"]

        for cur_axis in flip_axis_list:
            assert cur_axis in ["x", "y"]

            gt_boxes, points = getattr(augmentor_utils, "random_flip_along_{}".format(cur_axis))(
                gt_boxes=gt_boxes,
                points=points
            )

        return points, gt_boxes, gt_names

    def random_world_translation(self, points, gt_boxes, gt_names):
        """
        Translate world randomly

        :param points: (N, 4), 4 indicates (x, y, z, intensity)
        :param gt_boxes: (N, 7), 7 indicates (x, y, z, l, w, h, orientation)
        :param gt_names: (N, ), string
        :return:
        """
        along_axis_list = self._augmentation_config["RANDOM_WORLD_TRANSLATION"]["ALONG_AXIS_LIST"]
        world_translation_range = self._augmentation_config["RANDOM_WORLD_TRANSLATION"]["WORLD_TRANSLATION_RANGE"]

        for cur_axis in along_axis_list:
            assert cur_axis in ["x", "y", "z"]
            gt_boxes, points = getattr(augmentor_utils, "random_translation_along_{}".format(cur_axis))(
                gt_boxes=gt_boxes,
                points=points,
                offset_std=world_translation_range
            )

        return points, gt_boxes, gt_names

    def random_world_rotation(self, points, gt_boxes, gt_names):
        """
        Rotate world randomly

        :param points: (N, 4), 4 indicates (x, y, z, intensity)
        :param gt_boxes: (N, 7), 7 indicates (x, y, z, l, w, h, orientation)
        :param gt_names: (N, ), string
        :return:
        """
        rotation_range = self._augmentation_config["RANDOM_WORLD_ROTATION"]["WORLD_ROT_ANGLE"]
        gt_boxes, points = augmentor_utils.global_rotation(
            gt_boxes=gt_boxes,
            points=points,
            rot_range=rotation_range
        )
        return points, gt_boxes, gt_names

    def random_world_scaling(self, points, gt_boxes, gt_names):
        """
        Scale world randomly

        :param points: (N, 4), 4 indicates (x, y, z, intensity)
        :param gt_boxes: (N, 7), 7 indicates (x, y, z, l, w, h, orientation)
        :param gt_names: (N, ), string
        :return:
        """
        scale_range = self._augmentation_config["RANDOM_WORLD_SCALING"]["RANDOM_WORLD_SCALING_RANGE"]
        gt_boxes, points = augmentor_utils.global_scaling(
            gt_boxes=gt_boxes,
            points=points,
            scale_range=scale_range
        )
        return points, gt_boxes, gt_names

    def random_local_translation(self, points, gt_boxes, gt_names):
        """
        Local translation along x/y/z axis

        :param points: (N, 4), 4 indicates (x, y, z, intensity)
        :param gt_boxes: (N, 7), 7 indicates (x, y, z, l, w, h, orientation)
        :param gt_names: (N, ), string
        :return:
        """
        along_axis_list = self._augmentation_config["RANDOM_LOCAL_TRANSLATION"]["ALONG_AXIS_LIST"]
        local_translation_range = self._augmentation_config["RANDOM_LOCAL_TRANSLATION"]["LOCAL_TRANSLATION_RANGE"]

        for cur_axis in along_axis_list:
            assert cur_axis in ["x", "y", "z"]
            gt_boxes, points = getattr(augmentor_utils, "random_local_translation_along_{}".format(cur_axis))(
                gt_boxes=gt_boxes,
                points=points,
                offset_range=local_translation_range
            )

        return points, gt_boxes, gt_names

    def random_local_rotation(self, points, gt_boxes, gt_names):
        """
        Local rotation randomly

        :param points: (N, 4), 4 indicates (x, y, z, intensity)
        :param gt_boxes: (N, 7), 7 indicates (x, y, z, l, w, h, orientation)
        :param gt_names: (N, ), string
        :return:
        """
        rot_range = self._augmentation_config["RANDOM_LOCAL_ROTATION"]["LOCAL_ROT_ANGLE"]
        gt_boxes, points = augmentor_utils.local_rotation(
            gt_boxes=gt_boxes,
            points=points,
            rot_range=rot_range
        )

        return points, gt_boxes, gt_names

    def random_local_scaling(self, points, gt_boxes, gt_names):
        """
        Local scaling randomly

        :param points: (N, 4), 4 indicates (x, y, z, intensity)
        :param gt_boxes: (N, 7), 7 indicates (x, y, z, l, w, h, orientation)
        :param gt_names: (N, ), string
        :return:
        """
        scale_range = self._augmentation_config["RANDOM_LOCAL_SCALING"]["LOCAL_SCALE_RANGE"]
        gt_boxes, points = augmentor_utils.local_scaling(
            gt_boxes=gt_boxes,
            points=points,
            scale_range=scale_range
        )

        return points, gt_boxes, gt_names

    def forward(self, points, gt_boxes, category_names):
        """
        perform data augmentation

        :param points: (N, 4), 4 indicates (x, y, z, intensity)
        :param gt_boxes: (N, 7), 7 indicates (x, y, z, l, w, h, orientation)
        :param category_names: (N, ), string
        :return:
        """
        points, gt_boxes, gt_names = self.place_sampled_objects_on_current_points(
            raw_points=points,
            raw_gt_boxes=gt_boxes,
            raw_gt_names=category_names
        )

        points, gt_boxes, gt_names = self.random_world_flip(points=points, gt_boxes=gt_boxes, gt_names=gt_names)
        points, gt_boxes, gt_names = self.random_world_translation(points=points, gt_boxes=gt_boxes, gt_names=gt_names)
        points, gt_boxes, gt_names = self.random_world_rotation(points=points, gt_boxes=gt_boxes, gt_names=gt_names)
        points, gt_boxes, gt_names = self.random_world_scaling(points=points, gt_boxes=gt_boxes, gt_names=gt_names)
        points, gt_boxes, gt_names = self.random_local_translation(points=points, gt_boxes=gt_boxes, gt_names=gt_names)
        points, gt_boxes, gt_names = self.random_local_rotation(points=points, gt_boxes=gt_boxes, gt_names=gt_names)
        points, gt_boxes, gt_names = self.random_local_scaling(points=points, gt_boxes=gt_boxes, gt_names=gt_names)

        return points, gt_boxes, category_names


if __name__ == "__main__":
    from center_point_config import CenterPointConfig
    augmentor = PointsWithBoxes3DAugmentor(
        category_name_gt_lut_full_path=CenterPointConfig["DATASET_INFO"]["TRAIN_CATEGORY_GROUND_TRUTH_LUT_FULL_PATH"],
        augmentation_config=CenterPointConfig["DATA_AUGMENTATION_CONFIG"]
    )

