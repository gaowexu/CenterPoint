from utils import augmentor_utils
import numpy as np
import json


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

        self._sample_groups_config = self._augmentation_config["SAMPLING_GROUPS"]

        # build samples group look-up table (LUT)
        self._sample_groups = dict()
        for category_name, sample_num in self._sample_groups_config:
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

    def place_sampled_objects_on_current_points(self, points, gt_boxes, category_names):
        """
        place sampled ground truth objects on current lidar point cloud

        :param points: lidar points, np.ndarray, shape is (N, 4), 4 indicates (x, y, z, intensity)
        :param gt_boxes: original ground truth boxes, shape is (M, 7), 7 indicates (x, y, z, l, w, h, orientation)
        :param category_names: (M, ), string
        :return:
        """
        pass

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

