import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from voxel_generator import VoxelGenerator


class CenterPointDataset(Dataset):
    def __init__(self, voxel_size, class_names, point_cloud_range, phase, dataset_config):
        """
        数据集构造函数

        :param voxel_size: 体素分割大小，如 [0.16, 0.16, 4.0]
        :param class_names: 分类类别
        :param point_cloud_range: 点云范围
        :param phase: 阶段，"train" 或者 "val"
        :param dataset_config: 数据集配置
        """
        super(CenterPointDataset, self).__init__()
        self._phase = phase
        self._dataset_config = dataset_config
        self._voxel_size = voxel_size
        self._class_names = class_names
        self._point_cloud_range = point_cloud_range

        self._dataset_root_dir = self._dataset_config["ROOT_DIR"]
        self._max_num_points_per_voxel = self._dataset_config["MAX_NUM_POINTS_PER_VOXEL"]
        self._max_num_voxels = self._dataset_config["MAX_NUM_VOXELS"][phase]

        # 创建体素化转化器
        self._voxel_generator = VoxelGenerator(
            voxel_size=self._voxel_size,
            point_cloud_range=self._point_cloud_range,
            max_num_points_per_voxel=self._max_num_points_per_voxel,
            max_num_voxels=self._max_num_voxels
        )

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
            ret_samples.append({
                "lidar_data": os.path.join(self._dataset_root_dir, "lidar_data/{}.npy".format(sample_name)),
                "ground_truth": os.path.join(self._dataset_root_dir, "ground_truth/{}.json".format(sample_name))
            })
        return ret_samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        lidar_data_full_path = self._samples[index]["lidar_data"]
        gt_data_full_path = self._samples[index]["ground_truth"]

        points_cloud = np.load(lidar_data_full_path)
        ground_truth = json.load(open(gt_data_full_path, "r"))

        ret_data = {
            "points_cloud": points_cloud,
            "ground_truth": ground_truth
        }

        return ret_data

    def get_3d_gt_boxes(self, ground_truth):
        """
        从标注中提取3d box的标注信息

        :param ground_truth: dict, 数据集中的某一个样本的标注
        :return:
        """
        gt_boxes = list()
        for label in ground_truth:
            category = label["type"]
            if category not in self._class_names:
                continue

            # 这里做了+1处理，将0留给了背景类
            category_id = self._class_names.index(category) + 1
            x, y, z, length, width, height, orientation = \
                label["bbox"][0], label["bbox"][1], label["bbox"][2], \
                label["bbox"][3], label["bbox"][4], label["bbox"][5], \
                label["bbox"][6]

            # 类别信息放在box的最后
            gt_boxes.append([x, y, z, length, width, height, orientation, category_id])

        gt_boxes = np.array(gt_boxes)
        return torch.from_numpy(gt_boxes)

    def collate_batch(self, batch_list, _unused=True):
        batch_voxels_list = list()
        batch_indices_list = list()
        batch_sample_indices_list = list()
        batch_nums_per_voxel_list = list()

        batch_gt_3d_boxes_list = list()

        for index, sample in enumerate(batch_list):
            points = sample["points_cloud"]
            ground_truth = sample["ground_truth"]
            voxels, indices, num_per_voxel, _ = self._voxel_generator.generate(points=points)

            # 收集一个batch中每一个样本的ground truth
            gt_boxes = self.get_3d_gt_boxes(ground_truth=ground_truth)
            batch_gt_3d_boxes_list.append(gt_boxes)

            # 收集一个batch中的样本经过体素化之后的输出，包括voxels, indices, nums_per_voxel, sample_indices
            batch_voxels_list.append(voxels)
            batch_indices_list.append(indices)
            batch_nums_per_voxel_list.append(num_per_voxel)
            batch_sample_indices_list.append(index * torch.ones(voxels.shape[0]))

        # 在第一个维度上进行拼接，第一个维度拼接后根据batch_sample_indices可以知道属于batch中哪一个样本
        batch_voxels = torch.cat(batch_voxels_list, dim=0)
        batch_indices = torch.cat(batch_indices_list, dim=0)
        batch_nums_per_voxel = torch.cat(batch_nums_per_voxel_list, dim=0)
        batch_sample_indices = torch.cat(batch_sample_indices_list, dim=0)

        return batch_voxels, batch_indices, batch_nums_per_voxel, batch_sample_indices, batch_gt_3d_boxes_list


if __name__ == "__main__":
    from config import CenterPointConfig
    train_dataset = CenterPointDataset(
        voxel_size=CenterPointConfig["VOXEL_SIZE"],
        class_names=CenterPointConfig["CLASS_NAMES"],
        point_cloud_range=CenterPointConfig["POINT_CLOUD_RANGE"],
        phase="train",
        dataset_config=CenterPointConfig["DATASET_CONFIG"]
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        sampler=None,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_dataset.collate_batch
    )

    for batch_index, (batch_voxels, batch_indices, batch_nums_per_voxel,
                      batch_sample_indices, batch_gt_3d_boxes_list) in enumerate(train_dataloader):
        print("Batch index = {}".format(batch_index))
        print("batch_voxels.shape = {}".format(batch_voxels.shape))
        print("batch_indices.shape = {}".format(batch_indices.shape))
        print("batch_nums_per_voxel.shape = {}".format(batch_nums_per_voxel.shape))
        print("batch_sample_indices.shape = {}".format(batch_sample_indices.shape))

        print("\n")

    print(len(train_dataloader))
