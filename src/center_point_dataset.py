import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from voxel_generator import VoxelGenerator
from data_augmentor import PointsWithBoxes3DAugmentor
import open3d


class CenterPointDataset(Dataset):
    def __init__(self, voxel_size, class_names, point_cloud_range, phase,
                 dataset_config, augmentation_config, dataset_info_config):
        """
        数据集构造函数

        :param voxel_size: 体素分割大小，如 [0.16, 0.16, 4.0]
        :param class_names: 分类类别
        :param point_cloud_range: 点云范围
        :param phase: 阶段，"train" 或者 "val"
        :param dataset_config: 数据集配置
        :param augmentation_config: configuration of augmentation functions
        :param dataset_info_config: pre-calculated info configuration configuration
        """
        super(CenterPointDataset, self).__init__()
        self._phase = phase
        self._dataset_config = dataset_config
        self._augmentation_config = augmentation_config
        self._dataset_info_config = dataset_info_config
        self._voxel_size = voxel_size
        self._class_names = class_names
        self._point_cloud_range = point_cloud_range

        self._dataset_root_dir = self._dataset_config["RAW_DATASET_ROOT_DIR"]
        self._max_num_points_per_voxel = self._dataset_config["MAX_NUM_POINTS_PER_VOXEL"]
        self._max_num_voxels = self._dataset_config["MAX_NUM_VOXELS"][phase]

        # 创建体素化转化器
        self._voxel_generator = VoxelGenerator(
            voxel_size=self._voxel_size,
            point_cloud_range=self._point_cloud_range,
            max_num_points_per_voxel=self._max_num_points_per_voxel,
            max_num_voxels=self._max_num_voxels
        )

        # create data augmentation handler
        self._augmentor = PointsWithBoxes3DAugmentor(
            category_name_gt_lut_full_path=self._dataset_config["TRAIN_CATEGORY_GT_LUT_FULL_PATH"],
            augmentation_config=self._augmentation_config)

        # 获取所有的样本数据
        self._samples = self.collect_samples()

    def collect_samples(self):
        assert self._phase in ["train", "val"]
        if self._phase == "train":
            samples = open(os.path.join(self._dataset_root_dir, "splits/train.txt"), 'r').readlines()
        else:
            samples = open(os.path.join(self._dataset_root_dir, "splits/val.txt"), 'r').readlines()

        ret_samples = list()
        for sample_name in samples:
            sample_name = sample_name.strip()

            if self._phase == "train":
                ground_truth_full_path = self._dataset_info_config["TRAIN_SAMPLES_LABEL_ROOT_DIR"]
            else:
                ground_truth_full_path = self._dataset_info_config["VAL_SAMPLES_LABEL_ROOT_DIR"]

            gts = json.load(open(os.path.join(ground_truth_full_path, "{}.json".format(sample_name)), "r"))
            if len(gts) == 0:
                continue

            ret_samples.append({
                "lidar_data": os.path.join(self._dataset_root_dir, "lidar_data/{}.npy".format(sample_name)),
                "ground_truth": os.path.join(ground_truth_full_path, "{}.json".format(sample_name))
            })
        return ret_samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        lidar_data_full_path = self._samples[index]["lidar_data"]
        gt_data_full_path = self._samples[index]["ground_truth"]

        points = np.load(lidar_data_full_path)
        label_dict = json.load(open(gt_data_full_path, "r"))
        gt_boxes, category_names = self.get_gt_boxes_and_category_names(label_dict=label_dict)

        if self._phase == "train":
            points, gt_boxes, category_names = self._augmentor.forward(
                points=points,
                gt_boxes=gt_boxes,
                category_names=category_names
            )

        ret_data = {
            "points": points,
            "gt_boxes": gt_boxes,
            "category_names": category_names
        }

        return ret_data

    def get_gt_boxes_and_category_names(self, label_dict):
        """
        Get gt_box and category label from annotation

        :param label_dict: dict, ground truth
        :return:
        """
        gt_boxes = list()
        category_names = list()
        for label in label_dict:
            category = label["category"]
            if category not in self._class_names:
                continue

            x, y, z, length, width, height, orientation = \
                label["box3d"][0], label["box3d"][1], label["box3d"][2], \
                label["box3d"][3], label["box3d"][4], label["box3d"][5], \
                label["box3d"][6]

            gt_boxes.append([x, y, z, length, width, height, orientation])
            category_names.append(category)

        return np.array(gt_boxes), np.array(category_names)

    @staticmethod
    def translate_boxes_to_open3d_instance(gt_boxes):
        """
        Translate box with shape (7 + 1,) into open3d box instance
                 4-------- 6
               /|         /|
              5 -------- 3 .
              | |        | |
              . 7 -------- 1
              |/         |/
              2 -------- 0

        :param gt_boxes: ground truth box which indicates [x, y, z, l, w, h, orientation, category_id]
        :return:
        """
        center = gt_boxes[0:3]
        lwh = gt_boxes[3:6]
        axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # import ipdb; ipdb.set_trace(context=20)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        line_set.lines = open3d.utility.Vector2iVector(lines)
        return line_set, box3d

    @staticmethod
    def vis_points_cloud_with_gt_boxes(points, gt_boxes):
        """
        Plot 3D bounding boxes in Lidar coordinate system

        :param points: (N, 3+C_in)
        :param gt_boxes: (Q, 7)
        :return: None
        """
        vis = open3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.zeros(3)

        # draw origin
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(points[:, 0:3])
        vis.add_geometry(point_cloud)

        for i in range(gt_boxes.shape[0]):
            line_set, box3d = CenterPointDataset.translate_boxes_to_open3d_instance(gt_boxes[i])
            line_set.paint_uniform_color((1, 0, 0))
            vis.add_geometry(line_set)

        vis.run()
        vis.destroy_window()

    def append_category_id_to_gt_boxes(self, gt_boxes, category_names):
        """
        Append category id (plus 1) to the last dimension of gt_boxes

        :param gt_boxes: numpy.ndarray, shape = (N, 7), 7 indicates [x, y, z, l, w, h, orientation]
        :param category_names: numpy.ndarray, shape = (N, ), each element is string
        :return: torch.tensor, (N, 8)
        """
        assert gt_boxes.shape[0] == category_names.shape[0]

        # plus 1 because the index 0 is for background
        category_indices = np.array([self._class_names.index(cls_name) + 1 for cls_name in category_names])
        category_indices = category_indices[:, np.newaxis]
        return torch.from_numpy(np.concatenate((gt_boxes, category_indices), axis=-1))

    def collate_batch(self, batch_list, _unused=True):
        """
        Collect batch data for model training

        :param batch_list: a batch of samples
        :param _unused: defaults to True
        :return:
        """
        batch_voxels_list = list()
        batch_indices_list = list()
        batch_sample_indices_list = list()
        batch_nums_per_voxel_list = list()

        batch_gt_boxes_with_cls_id_list = list()

        for index, sample in enumerate(batch_list):
            # the points, gt_boxes, categories are already augmented in __getitem__() function
            points = sample["points"]
            gt_boxes = sample["gt_boxes"]
            category_names = sample["category_names"]

            # append category id (plus 1) to the end of gt_boxes
            gt_boxes_with_cls_id = self.append_category_id_to_gt_boxes(gt_boxes, category_names)

            # self.vis_points_cloud_with_gt_boxes(points=points, gt_boxes=gt_boxes_with_cls_id)

            # perform point cloud voxelization
            voxels, indices, num_per_voxel, _ = self._voxel_generator.generate(points=points)
            batch_gt_boxes_with_cls_id_list.append(gt_boxes_with_cls_id)

            # collects the output of voxelization over all samples in current batch
            batch_voxels_list.append(voxels)
            batch_indices_list.append(indices)
            batch_nums_per_voxel_list.append(num_per_voxel)
            batch_sample_indices_list.append(index * torch.ones(voxels.shape[0]))

        # concat along first dimension
        batch_voxels = torch.cat(batch_voxels_list, dim=0)
        batch_indices = torch.cat(batch_indices_list, dim=0)
        batch_nums_per_voxel = torch.cat(batch_nums_per_voxel_list, dim=0)
        batch_sample_indices = torch.cat(batch_sample_indices_list, dim=0)

        return batch_voxels, batch_indices, batch_nums_per_voxel, batch_sample_indices, batch_gt_boxes_with_cls_id_list


if __name__ == "__main__":
    from center_point_config import CenterPointConfig
    train_dataset = CenterPointDataset(
        voxel_size=CenterPointConfig["VOXEL_SIZE"],
        class_names=CenterPointConfig["CLASS_NAMES"],
        point_cloud_range=CenterPointConfig["POINT_CLOUD_RANGE"],
        phase="train",
        dataset_config=CenterPointConfig["DATASET_CONFIG"],
        augmentation_config=CenterPointConfig["DATA_AUGMENTATION_CONFIG"],
        dataset_info_config=CenterPointConfig["DATASET_CONFIG"]
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        sampler=None,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=5,
        collate_fn=train_dataset.collate_batch
    )

    for batch_index, (voxels, indices, nums_per_voxel,
                      sample_indices, gt_boxes_with_cls_id_list) in enumerate(train_dataloader):
        print("Batch index = {}".format(batch_index))
        print("batch_voxels.shape = {}".format(voxels.shape))
        print("batch_indices.shape = {}".format(indices.shape))
        print("batch_nums_per_voxel.shape = {}".format(nums_per_voxel.shape))
        print("batch_sample_indices.shape = {}".format(sample_indices.shape))

        print("\n")

    print(len(train_dataloader))
