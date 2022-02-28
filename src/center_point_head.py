import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import utils.centernet_utils as centernet_utils
import utils.model_nms_utils as model_nms_utils


class SeparateHead(nn.Module):
    def __init__(self, input_channels, separate_head_dict, init_bias, use_bias):
        """
        子任务回归预测头

        :param input_channels: 输入的通道数，即 point-pillar 2D backbone的输出channels
        :param separate_head_dict: 子任务检测头的配置字典
        :param init_bias: heatmap(center-ness score map)卷积层初始化时的偏置初始化值
        :param use_bias: 卷积层（不含最后一个与任务相关的卷积）是否启动偏置
        """
        super(SeparateHead, self).__init__()
        self._input_channels = input_channels
        self._separate_head_dict = separate_head_dict
        self._init_bias = init_bias
        self._use_bias = use_bias

        for key_name in self._separate_head_dict.keys():
            # 获取没一个子任务（center, center_z, dim, rot, hm）的卷积输出通道数
            output_channels = self._separate_head_dict[key_name]["out_channels"]

            # 获取卷积层数量，包含最后一层与回归任务相关的卷积
            num_conv = self._separate_head_dict[key_name]["num_conv"]

            conv_layers = list()
            for k in range(num_conv - 1):
                conv_layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=input_channels,
                            out_channels=input_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1,
                            bias=use_bias),
                        nn.BatchNorm2d(input_channels),
                        nn.ReLU()
                    )
                )

            # 添加任务相关的最后一层卷积层
            conv_layers.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                    bias=True)
            )

            sub_task_layers = nn.Sequential(*conv_layers)

            # 神经网络模块的权重初始化
            if key_name == "hm":
                sub_task_layers[-1].bias.data.fill_(init_bias)
            else:
                for m in sub_task_layers.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)

                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)

            # 设置为属性
            self.__setattr__(key_name, sub_task_layers)

    def forward(self, x):
        """
         子任务推理函数

        :param x: torch.Tensor, 形状为 (batch_size, 6C, nx/2, ny/2)
        :return:
        """
        # center, center_z, dim, rot, hm
        center_out = self.__getattr__("center")(x)
        center_z_out = self.__getattr__("center_z")(x)
        dim_out = self.__getattr__("dim")(x)
        rot_out = self.__getattr__("rot")(x)
        hm_out = self.__getattr__("hm")(x)

        return center_out, center_z_out, dim_out, rot_out, hm_out


class CenterBBoxHead(nn.Module):
    def __init__(self,
                 input_channels,
                 point_cloud_range,
                 voxel_size,
                 class_names,
                 head_config,
                 target_assigner_config,
                 post_processing_config):
        super(CenterBBoxHead, self).__init__()

        self._input_channels = input_channels
        self._point_cloud_range = point_cloud_range
        self._voxel_size = voxel_size
        self._class_names = class_names
        self._head_config = head_config
        self._target_assigner_config = target_assigner_config
        self._post_processing_config = post_processing_config

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
        self._separate_head_cfg = self._head_config["separate_head_config"]
        for index, single_head_class_names in enumerate(self._class_names_each_head):
            head_dict = copy.deepcopy(self._separate_head_cfg["head_dict"])

            # 添加 heatmap (center-ness score map) 的网络配置
            head_dict["hm"] = {
                "out_channels": len(single_head_class_names),
                "num_conv": self._head_config["num_hm_conv"]
            }

            # 构建独立任务的检测头
            self._heads_list.append(
                SeparateHead(
                    input_channels=self._head_config["shared_conv_channel"],
                    separate_head_dict=head_dict,
                    init_bias=-2.19,
                    use_bias=self._head_config.get("use_bias_before_norm", False)
                )
            )

        # 保存每一次 forward 结果
        self._forward_ret_dict = dict()

    def generate_predicted_boxes(self, batch_size, predictions_list):
        """
        生成预测框

        :param batch_size: batch大小
        :param predictions_list: 预测结果，是一个list，分别对应着 separate branch的预测结果
        :return:
        """
        post_center_limit_range = torch.tensor(self._post_processing_config["post_center_limit_range"]).cuda().float()

        ret_dict = [{
            "pred_boxes": list(),
            "pred_scores": list(),
            "pred_labels": list()
        } for _ in range(batch_size)]

        for index, predictions in enumerate(predictions_list):
            # batch_hm's shape = torch.Size([4, Q, 216, 248]), Q 为当前分支负责预测回归的类别数目
            batch_hm = predictions['hm'].sigmoid()
            batch_center = predictions['center']                            # shape = torch.Size([4, 2, 216, 248])
            batch_center_z = predictions['center_z']                        # shape = torch.Size([4, 1, 216, 248])
            batch_dim = predictions['dim'].exp()                            # shape = torch.Size([4, 3, 216, 248])
            batch_rot_cos = predictions['rot'][:, 0].unsqueeze(dim=1)       # shape = torch.Size([4, 1, 216, 248])
            batch_rot_sin = predictions['rot'][:, 1].unsqueeze(dim=1)       # shape = torch.Size([4, 1, 216, 248])

            final_predictions = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm,
                rot_cos=batch_rot_cos,
                rot_sin=batch_rot_sin,
                center=batch_center,
                center_z=batch_center_z,
                dim=batch_dim,
                vel=None,
                point_cloud_range=self._point_cloud_range,
                voxel_size=self._voxel_size,
                feature_map_stride=self._target_assigner_config["feature_map_stride"],
                K=self._post_processing_config["max_objs_per_sample"],
                circle_nms=(self._post_processing_config["nms_type"] == "circle_nms"),
                score_thresh=self._post_processing_config["score_thresh"],
                post_center_limit_range=post_center_limit_range
            )

            print("final_predictions = {}".format(final_predictions))

            for batch_index, final_dict in enumerate(final_predictions):
                final_dict["pred_labels"] = self._class_id_mapping_each_head[index][final_dict["pred_labels"].long()]
                if self._post_processing_config["nms_type"] != "circle_nms":
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict["pred_scores"],
                        box_preds=final_dict["pred_boxes"],
                        nms_config=self._head_con
                    )


                ret_dict[batch_index]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[batch_index]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[batch_index]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, predicted_boxes):
        """

        :param batch_size:
        :param predicted_boxes:
        :return:
        """
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in predicted_boxes])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = predicted_boxes[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for batch_index in range(batch_size):
            num_boxes = len(predicted_boxes[batch_index]['pred_boxes'])

            rois[batch_index, :num_boxes, :] = predicted_boxes[batch_index]['pred_boxes']
            roi_scores[batch_index, :num_boxes] = predicted_boxes[batch_index]['pred_scores']
            roi_labels[batch_index, :num_boxes] = predicted_boxes[batch_index]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, x):
        """
        CenterPoint 检测头推理函数

        :param x: torch.Tensor, 形状为 (batch_size, 6C, nx/2, ny/2)
        :return:
        """
        # shared_feats输出维度为 (batch_size, 64, 216, 248)
        shared_feats = self._shared_conv(x)
        batch_size = shared_feats.shape[0]

        # 举例说明：如 class_names_each_head = [["Vehicle", "Pedestrian"], ["Cyclist"]]， 则说明分为两个branch来预测，
        # 第一个branch负责检测vehicle和pedestrian, 第二个branch负责检测cyclist, 故而下述predictions变量中存储的是两个元素，
        # 每一个元素包含 center, center_z, dim, rot, hm的预测tensor
        predictions_list = list()
        for head in self._heads_list:
            center_out, center_z_out, dim_out, rot_out, hm_out = head(shared_feats)
            predictions_list.append(
                {
                    "center": center_out,
                    "center_z": center_z_out,
                    "dim": dim_out,
                    "rot": rot_out,
                    "hm": hm_out
                }
            )

        # 生成预测框
        predicted_boxes = self.generate_predicted_boxes(batch_size=batch_size, predictions_list=predictions_list)

        rois, roi_scores, roi_labels = self.reorder_rois_for_refining(
            batch_size=batch_size,
            predicted_boxes=predicted_boxes)

        return rois, roi_scores, roi_labels


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
        },
        target_assigner_config={
            "feature_map_stride": 8,
            "num_max_objs": 500,
            "gaussian_overlap": 0.1,
            "min_radius": 2,
        },
        post_processing_config={
            "score_thresh": 0.1,
            "post_center_limit_range": [0, -39.68, -3, 69.12, 39.68, 1],
            "max_objs_per_sample": 500,
            "nms_type": "nms_gpu",
            "nms_thresh": 0.70,
            "nms_pre_max_size": 4096,
            "nms_post_max_size": 500
        }
    )

    x = torch.rand((4, 384, 216, 248))

    bbox_head(x)


