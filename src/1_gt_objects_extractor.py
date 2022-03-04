import os
import json
import numpy as np
import torch
import sys
sys.path.append("../")
from ops.roiaware_pool3d import roiaware_pool3d_utils


class LidarGroundTruthObjectsExtractor(object):
    def __init__(self, dataset_root_dir, db_info_config):
        """
        Sampling the point clouds for ground truth objects and save them into local disk,
        which will be used for data augmentation before model training

        :param dataset_root_dir: root directory of dataset
        :param db_info_config: dataset info configuration
        """
        self._dataset_root_dir = dataset_root_dir
        self._db_info_config = db_info_config
        self._objects_points_cloud_saving_dir = self._db_info_config["GROUND_TRUTH_ROOT_DIR"]
        self._train_category_gt_lut_full_path = self._db_info_config["TRAIN_CATEGORY_GROUND_TRUTH_LUT_FULL_PATH"]
        self._train_samples_label_root_dir = self._db_info_config["TRAIN_SAMPLES_LABEL_ROOT_DIR"]
        self._val_samples_label_root_dir = self._db_info_config["VAL_SAMPLES_LABEL_ROOT_DIR"]

        if not os.path.exists(self._objects_points_cloud_saving_dir):
            os.makedirs(self._objects_points_cloud_saving_dir)

        if not os.path.exists(self._train_samples_label_root_dir):
            os.makedirs(self._train_samples_label_root_dir)

        if not os.path.exists(self._val_samples_label_root_dir):
            os.makedirs(self._val_samples_label_root_dir)

        self._lidar_data_root_dir = os.path.join(self._dataset_root_dir, "lidar_data")
        self._ground_truth_root_dir = os.path.join(self._dataset_root_dir, "ground_truth")
        self._split_train_samples_full_path = os.path.join(self._dataset_root_dir, "splits/train.txt")
        self._split_val_samples_full_path = os.path.join(self._dataset_root_dir, "splits/val.txt")
        self._train_sample_names = [
            name.strip() for name in open(self._split_train_samples_full_path, "r").readlines() if name.strip()]
        self._val_sample_names = [
            name.strip() for name in open(self._split_val_samples_full_path, "r").readlines() if name.strip()]

    @staticmethod
    def get_object_difficulty(box2d, truncation, occlusion):
        """
        get object difficulty based on KITTI dataset standard

        :param box2d:
        :param truncation:
        :param occlusion:
        :return:
        """
        height = float(box2d[3]) - float(box2d[1]) + 1
        if height >= 40 and truncation <= 0.15 and occlusion <= 0:
            level_str = 'Easy'
            return 0
        elif height >= 25 and truncation <= 0.3 and occlusion <= 1:
            level_str = 'Moderate'
            return 1
        elif height >= 25 and truncation <= 0.5 and occlusion <= 2:
            level_str = 'Hard'
            return 2
        else:
            level_str = 'UnKnown'
            return -1

    def extract_gt_samples(self, phase="train"):
        """
        Extract ground truth objects and saving them into local disk for further augmentation

        :param phase: "train" or "val"

        :return:
        """
        assert phase in ["train", "val"]
        samples_names = self._train_sample_names if phase == "train" else self._val_sample_names

        samples_amount = len(samples_names)
        category_gt_lut = dict()

        for index, sample_name in enumerate(samples_names):
            print("Exacting sample {} ({}/{}) in training subset...".format(sample_name, index+1, samples_amount))

            points = np.load(os.path.join(self._lidar_data_root_dir, "{}.npy".format(sample_name)))
            gts = json.load(open(os.path.join(self._ground_truth_root_dir, "{}.json".format(sample_name)), "r"))

            gt_boxes = np.array([label["bbox"] for label in gts])

            # point_masks's shape = (objs_amount, len(points)), point_masks[i][j] = 1 represents whether the j-th
            # cloud point is inside the i-th ground truth box
            point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(
                points=torch.from_numpy(points[:, 0:3]),
                boxes=torch.from_numpy(gt_boxes)
            ).numpy()

            gt_boxes_info = list()
            for obj_idx, label in enumerate(gts):
                gt_points = points[point_masks[obj_idx] > 0]

                # minus the cloud points of a given object with its center (x_c, y_c, z_c) and then it is
                # position independent object and further could be placed in anywhere on the ground plane
                gt_points[:, :3] -= gt_boxes[obj_idx, :3]

                if phase == "train":
                    dump_full_path = os.path.join(
                        self._objects_points_cloud_saving_dir,
                        "sample_{}_category_{}_{}.npy".format(sample_name, label["type"], obj_idx)
                    )
                    np.save(dump_full_path, gt_points)
                else:
                    dump_full_path = None

                difficulty = self.get_object_difficulty(
                    box2d=label["box2d"],
                    truncation=label["truncation"],
                    occlusion=label["occlusion"]
                )

                category = label["type"]
                gt_sampled_object = {
                    "sample_name": sample_name,
                    "category": category,
                    "num_points_in_gt": len(gt_points),
                    "difficulty": difficulty,
                    "box3d": label["bbox"],
                    "path": os.path.abspath(dump_full_path) if dump_full_path is not None else None,
                    "gt_index": obj_idx
                }
                gt_boxes_info.append(gt_sampled_object)

                if phase == "train":
                    if category not in category_gt_lut.keys():
                        category_gt_lut[category] = [gt_sampled_object]
                    else:
                        category_gt_lut[category].append(gt_sampled_object)

                # save sample detail info into local disk
                samples_gt_dir = self._train_samples_label_root_dir if phase == "train" else self._val_samples_label_root_dir
                sample_gt_dump_full_path = os.path.join(samples_gt_dir, "{}.json".format(sample_name))
                json.dump(gt_boxes_info, open(sample_gt_dump_full_path, "w"), indent=True)

        if phase == "train":
            json.dump(category_gt_lut, open(self._train_category_gt_lut_full_path, "w"), indent=True)


if __name__ == "__main__":
    from center_point_config import CenterPointConfig

    objects_extractor = LidarGroundTruthObjectsExtractor(
        dataset_root_dir=CenterPointConfig["DATASET_CONFIG"]["ROOT_DIR"],
        db_info_config=CenterPointConfig["DATASET_INFO"]
    )

    objects_extractor.extract_gt_samples(phase="train")
    objects_extractor.extract_gt_samples(phase="val")
