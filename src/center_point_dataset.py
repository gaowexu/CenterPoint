import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from voxel_generator import VoxelGenerator


class CenterPointDataset(Dataset):
    def __init__(self, phase, dataset_config):
        super(CenterPointDataset, self).__init__()
        self._phase = phase
        self._dataset_config = dataset_config
        self._dataset_root_dir = self._dataset_config["root_dir"]
        self._voxel_size = self._dataset_config["voxel_size"]
        self._point_cloud_range = self._dataset_config["point_cloud_range"]
        self._max_num_points_per_voxel = self._dataset_config["max_num_points_per_voxel"]
        self._max_num_voxels = self._dataset_config["max_num_voxels"][phase]

        self._voxel_generator = VoxelGenerator(
            voxel_size=self._voxel_size,
            point_cloud_range=self._point_cloud_range,
            max_num_points_per_voxel=self._max_num_points_per_voxel,
            max_num_voxels=self._max_num_voxels
        )

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

    def collate_batch(self, batch_list, _unused=True):
        voxels_list = list()
        indices_list = list()
        sample_indices_list = list()
        nums_per_voxel_list = list()

        for index, sample in enumerate(batch_list):
            points = sample["points_cloud"]
            voxels, indices, num_per_voxel, _ = self._voxel_generator.generate(points=points)

            voxels_list.append(voxels)
            indices_list.append(indices)
            nums_per_voxel_list.append(num_per_voxel)
            sample_indices_list.append(index * torch.ones(voxels.shape[0]))

        batch_voxels = torch.cat(voxels_list, dim=0)
        batch_indices = torch.cat(indices_list, dim=0)
        batch_nums_per_voxel = torch.cat(nums_per_voxel_list, dim=0)
        batch_sample_indices = torch.cat(sample_indices_list, dim=0)

        return batch_voxels, batch_indices, batch_nums_per_voxel, batch_sample_indices


if __name__ == "__main__":
    train_dataset = CenterPointDataset(
        phase="train",
        dataset_config={
            "root_dir": "../dataset",
            "point_cloud_range": [0, -39.68, -3, 69.12, 39.68, 1.0],
            "voxel_size": [0.16, 0.16, 4.0],
            "max_num_points_per_voxel": 100,
            "max_num_voxels": {
                "train": 16000,
                "val": 40000,
            }
        }
    )
    print(len(train_dataset))

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        sampler=None,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_dataset.collate_batch
    )

    for batch_index, (batch_voxels, batch_indices, batch_nums_per_voxel, batch_sample_indices) in enumerate(train_dataloader):
        print("Batch index = {}".format(batch_index))
        print("batch_voxels.shape = {}".format(batch_voxels.shape))
        print("batch_indices.shape = {}".format(batch_indices.shape))
        print("batch_nums_per_voxel.shape = {}".format(batch_nums_per_voxel.shape))
        print("batch_sample_indices.shape = {}".format(batch_sample_indices.shape))

        print("\n")
        break

    print(len(train_dataloader))
