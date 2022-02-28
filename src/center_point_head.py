import copy
import numpy as np
import torch
import torch.nn as nn


class CenterBBoxHead(nn.Module):
    def __init__(self,
                 input_channels,
                 point_cloud_range,
                 voxel_size,
                 class_names,
                 head_config):
        super(CenterBBoxHead, self).__init__()

        self._input_channels = input_channels
        self._point_cloud_range = point_cloud_range
        self._voxel_size = voxel_size
        self._class_names = class_names
        self._head_config = head_config

        # 获取每一个head中的类别信息以及类别名和id的映射关系，如在检测中共需要检测 ['Vehicle', 'Pedestrian', 'Cyclist'] 三个
        # 类别的3D物体，且['Vehicle', 'Pedestrian']和['Cyclist']分两个branch进行检测，则两个branch在共享卷积层之后会分别回归
        # center, center_z, dim, rot, hm，其中第一个分支输出的hm的channel数为2（类别数), 第二个分支输出的 hm 的 channel 数
        # 为1.
        self._class_names_each_head = list()
        self._class_id_mapping_each_head = list()
        for single_head_class_names in self._head_config["class_names_each_head"]:
            self._class_names_each_head.append(single_head_class_names)

            class_id_mapping = torch.from_numpy(np.array([
                self._class_names.index(x) for x in single_head_class_names if x in self._class_names
            ]))
            self._class_id_mapping_each_head.append(class_id_mapping)

        assert sum([len(x) for x in self._class_names_each_head]) == len(self._class_names)

        # 构建共享卷积层
        self._shared_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=self._head_config["shared_conv_channel"],
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=self._head_config.get("use_bias_before_norm", False)
            ),
            nn.BatchNorm2d(self._head_config["shared_conv_channel"]),
            nn.ReLU()
        )

        # 构建 separate branch
        self._heads_list = nn.ModuleList()




if __name__ == "__main__":
    bbox_head = CenterBBoxHead(
        input_channels=384,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=[0.16, 0.16, 4.0],
        class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
        head_config={
            "class_names_each_head": [
                ["Vehicle", "Pedestrian"],
                ["Cyclist"]
            ],
            "shared_conv_channel": 64,
            "use_bias_before_norm": True,
            "num_hm_conv": 2,
            "separate_head_config": {
                "head_order": ["center", "center_z", "dim", "rot"],
                "head_dict": {
                    "center": {"out_channels": 2, "num_conv": 2},
                    "center_z": {"out_channels": 1, "num_conv": 2},
                    "dim": {"out_channels": 3, "num_conv": 2},
                    "rot": {"out_channels": 2, "num_conv": 2},
                },
            }
        }
    )

    print(bbox_head)

