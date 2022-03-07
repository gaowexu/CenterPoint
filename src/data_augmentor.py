from utils import augmentor_utils
import numpy as np
import json
from ops.iou3d_nms import iou3d_nms_utils


class PointsWithBoxes3DAugmentor(object):
    def __init__(self, category_name_gt_lut_full_path, augmentation_config):
        """
        Constructor

        :param category_name_gt_lut_full_path:
        :param augmentation_config:
        """
        self._category_name_gt_lut_full_path = category_name_gt_lut_full_path
        self._augmentation_config = augmentation_config

        # load ground truth samples pool from training dataset
        self._train_gt_boxes_pool = json.load(open(self._category_name_gt_lut_full_path, "r"))

        self._limit_whole_scene = self._augmentation_config["LIMIT_WHOLE_SCENE"]
        self._sample_groups_config = self._augmentation_config["SAMPLING_GROUPS"]

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
        points = raw_points
        if self._use_road_plane:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes=sampled_gt_boxes,
                road_plane=road_plane
            )

        obj_points_list = list()





        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points_aug = np.concatenate([obj_points, points], axis=0)
        gt_boxes_aug = np.concatenate([raw_gt_boxes, sampled_gt_boxes], axis=0)
        gt_names_aug = np.concatenate([raw_gt_names, sampled_gt_names], axis=0)

        return points_aug, gt_boxes_aug, gt_names_aug

    def random_world_flip(self, points, gt_boxes):
        flip_axis_list = self._augmentation_config["RANDOM_WORLD_FLIP_ALONG_AXIS_LIST"]

        for axis in flip_axis_list:
            assert axis in ["x", "y"]
            if axis == "x":
                gt_boxes, points = augmentor_utils.random_flip_along_x(
                    gt_boxes=gt_boxes,
                    points=points)
            else:
                gt_boxes, points = augmentor_utils.random_flip_along_y(
                    gt_boxes=gt_boxes,
                    points=points)

        return points, gt_boxes

    def random_world_rotation(self, points, gt_boxes):
        rotation_range = self._augmentation_config["RANDOM_WORLD_ROTATION_ANGLE"]
        assert isinstance(rotation_range, list)
        gt_boxes, points = augmentor_utils.global_rotation(
            gt_boxes=gt_boxes,
            points=points,
            rot_range=rotation_range
        )
        return points, gt_boxes

    def random_world_scaling(self, points, gt_boxes):
        scale_range = self._augmentation_config["RANDOM_WORLD_SCALING_RANGE"]
        assert isinstance(scale_range, list)

        gt_boxes, points = augmentor_utils.global_scaling(
            gt_boxes=gt_boxes,
            points=points,
            scale_range=scale_range
        )
        return points, gt_boxes

    def forward(self, points, gt_boxes, category_names):
        """
        perform data augmentation

        :param points: (N, 4), 4 indicates (x, y, z, intensity)
        :param gt_boxes: (N, 7), 7 indicates (x, y, z, l, w, h, orientation)
        :param category_names: (N, ), string
        :return:
        """
        points, gt_boxes = self.random_world_scaling(points=points, gt_boxes=gt_boxes)
        return points, gt_boxes, category_names


if __name__ == "__main__":
    from center_point_config import CenterPointConfig
    augmentor = PointsWithBoxes3DAugmentor(
        category_name_gt_lut_full_path=CenterPointConfig["DATASET_INFO"]["TRAIN_CATEGORY_GROUND_TRUTH_LUT_FULL_PATH"],
        augmentation_config=CenterPointConfig["DATA_AUGMENTATION_CONFIG"]
    )

