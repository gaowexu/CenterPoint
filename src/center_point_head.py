import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import utils.centernet_utils as centernet_utils
import utils.model_nms_utils as model_nms_utils
from utils.loss_utils import FocalLossCenterNet, RegLossCenterNet


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

        # 构建损失函数
        self.build_losses()

        # 保存每一次 forward 结果
        self._forward_ret_dict = dict()

    def build_losses(self):
        self.add_module("hm_loss_func", FocalLossCenterNet())
        self.add_module("reg_loss_func", RegLossCenterNet())

    @staticmethod
    def sigmoid(x):
        eps = 1e-4
        return torch.clamp(x.sigmoid(), min=eps, max=1.0 - eps)

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    @staticmethod
    def assign_target_of_single_head(
            point_cloud_range,
            voxel_size,
            num_classes,
            gt_boxes,
            feature_map_size,
            feature_map_stride,
            num_max_objs=500,
            gaussian_overlap=0.1,
            min_radius=2):
        """

        :param point_cloud_range: 点云的检测范围，[x_min, y_min, z_min, x_max, y_max, z_max]
        :param voxel_size: 体素的分割大小，[x_size, y_size, z_size]， 默认为 [0.16, 0.16, 4.0]
        :param num_classes: 某一个检测头负责检测的目标类别数目，如一个检测头负责检测 pedestrian 和 car 两个类别，则该值为 2
        :param gt_boxes: 某一个检测头负责检测的目标真值3D矩形狂，形状为(N, 8)
        :param feature_map_size: Point Pillar中体素化后二维 backbone 提取的特征维度，如KITTI数据集中大小为 (216, 248)
        :param feature_map_stride: 特征图相比于原始输入的时候降采样的比例，如KITTI数据集中为2
        :param num_max_objs: 默认单个检测头中最多的目标框数目
        :param gaussian_overlap:
        :param min_radius:
        :return:
        """
        # heatmap, ret_boxes, inds, masks的初始化，形状分别为 (num_classes, 248, 216), (500, 8), (500,), (500,)
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1]))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        # gt_boxes的形状为(N, 8), 8代表着 x, y, z, dx, dy, dz, orientation, category_id_in_separate_single_head
        # 其中 dx, dy, dz 也可表示为物体的 length, width, height
        x, y, z, dx, dy, dz = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[
                                                                                                              :, 5]

        coord_x = (x - point_cloud_range[0]) / voxel_size[0] / feature_map_stride
        coord_y = (y - point_cloud_range[1]) / voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        # 获取三维物体在feature map (PointPillar的2D backbone输出)上的长度(dx)和宽度(dy)
        dx = dx / voxel_size[0] / feature_map_stride
        dy = dy / voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            # 如果物体的长度或宽度在feature map中尺度小于等于0, 则忽略
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            # 如果物体的中心点落在 feature map 的外边，则忽略
            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            # 获取第k个矩形框的类别id，该id是当前separate head中的类别索引，id是存储在box的最后一维，减1是因为在之前构造gt_boxes
            # 时加上了1 (加1是因为把0留给了背景类)
            cls_id_in_curr_separate_head = (gt_boxes[k, -1] - 1).long()

            centernet_utils.draw_gaussian_to_heatmap(heatmap[cls_id_in_curr_separate_head], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None):

        ret_dict = {
            "heatmaps": list(),
            "target_boxes": list(),
            "inds": list(),
            "masks": list(),
            "heatmap_masks": list()
        }

        all_names = np.array(["bg", *self._class_names])
        batch_size = len(gt_boxes)

        for index, single_head_class_names in enumerate(self._class_names_each_head):
            target_hm_list, target_boxes_list, target_inds_list, target_masks_list = list(), list(), list(), list()

            for batch_index in range(batch_size):
                gt_boxes_in_curr_sample = gt_boxes[batch_index]
                gt_class_names_in_curr_sample = all_names[gt_boxes_in_curr_sample[:, 0].cpu().long().numpy()]

                gt_boxes_single_head = list()

                for box_index, name in enumerate(gt_class_names_in_curr_sample):
                    if name not in single_head_class_names:
                        continue

                    box = gt_boxes_in_curr_sample[box_index]
                    box[0] = single_head_class_names.index(name) + 1
                    gt_boxes_single_head.append(box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = gt_boxes_in_curr_sample[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, boxes, inds, mask = self.assign_target_of_single_head(
                    point_cloud_range=self._point_cloud_range,
                    voxel_size=self._voxel_size,
                    num_classes=len(single_head_class_names),
                    gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size,
                    feature_map_stride=self._target_assigner_config["feature_map_stride"],
                    num_max_objs=self._target_assigner_config["num_max_objs"],
                    gaussian_overlap=self._target_assigner_config["gaussian_overlap"],
                    min_radius=self._target_assigner_config["min_radius"],
                )

                target_hm_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(boxes.to(gt_boxes_single_head.device))
                target_inds_list.append(inds.to(gt_boxes_single_head.device))
                target_masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(target_hm_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(target_inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(target_masks_list, dim=0))

        return ret_dict

    def generate_predicted_boxes(self, batch_size, predictions_list):
        """
        融合 separate head 的预测值，生成预测框

        :param batch_size: batch大小
        :param predictions_list: 预测结果，是一个list，分别对应着 separate branch 的预测结果
        :return:
        """
        ret_dict = [{
            'pred_boxes': list(),
            'pred_scores': list(),
            'pred_labels': list(),      # 存储的是类别id
        } for _ in range(batch_size)]

        for separate_head_index, predictions in enumerate(predictions_list):
            batch_hm = predictions['hm'].sigmoid()
            batch_center = predictions['center']
            batch_center_z = predictions['center_z']
            batch_dim = predictions['dim'].exp()
            batch_rot_cos = predictions['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = predictions['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm,
                rot_cos=batch_rot_cos,
                rot_sin=batch_rot_sin,
                center=batch_center,
                center_z=batch_center_z,
                dim=batch_dim,
                vel=batch_vel,
                point_cloud_range=self._point_cloud_range,
                voxel_size=self._voxel_size,
                feature_map_stride=self._target_assigner_config["feature_map_stride"],
                K=self._post_processing_config["max_objs_per_sample"],
                circle_nms=(self._post_processing_config["nms_type"] == 'circle_nms'),
                score_thresh=self._post_processing_config["score_thresh"],
                post_center_limit_range=torch.tensor(self._post_processing_config["post_center_limit_range"]).cuda().float()
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self._class_id_mapping_each_head[separate_head_index][final_dict['pred_labels'].long()]
                if self._post_processing_config["nms_type"] != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'],
                        box_preds=final_dict['pred_boxes'],
                        nms_type=self._post_processing_config["nms_type"],
                        nms_thresh=self._post_processing_config["nms_thresh"],
                        nms_pre_max_size=self._post_processing_config["nms_pre_max_size"],
                        nms_post_max_size=self._post_processing_config["nms_post_max_size"],
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, predictions):
        """
        找出当前batch中样本预测得到的物体数量，再以该数量规整化所有的样本，使得对于一个batch中输出的 rois, roi_scores,
        roi_labels的第一维度一致，即预测的物体数量一致

        :param batch_size:
        :param predictions:
        :return:
        """
        # 找出当前batch中最大的预测物体数量
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in predictions])
        num_max_rois = max(1, num_max_rois)

        pred_boxes = predictions[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for batch_index in range(batch_size):
            valid_boxes_num = len(predictions[batch_index]['pred_boxes'])
            rois[batch_index, :valid_boxes_num, :] = predictions[batch_index]['pred_boxes']
            roi_scores[batch_index, :valid_boxes_num] = predictions[batch_index]['pred_scores']
            roi_labels[batch_index, :valid_boxes_num] = predictions[batch_index]['pred_labels']

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
        # 第一个branch负责检测vehicle和pedestrian, 第二个branch负责检测cyclist, 故而下述predictions_list变量中存储的是
        # 两个元素，每一个元素包含 center, center_z, dim, rot, hm的预测tensor
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

        # 生成预测框, 其类型为一个数组，数组的长度为 batch_size, 数组中每一个元素的类型为字典，包含 pred_boxes, pred_scores,
        # pred_labels 三个字段, 其中 pred_boxes的维度为(Q, 7), pred_scores的维度为 (Q,), pred_labels的维度为 (Q, ). Q为
        # 检测出来的三维目标数
        predictions = self.generate_predicted_boxes(batch_size=batch_size, predictions_list=predictions_list)

        # 对于batch中预测结果进行统一预测物体的数量，输出 rois, roi_scores, roi_labels 的维度分别为 (batch_size, X, 7),
        # (batch_size, X), (batch_size, X), 其中 X 为当前 batch 中样本预测得到的物体数量的最大值
        rois, roi_scores, roi_labels = self.reorder_rois_for_refining(
            batch_size=batch_size,
            predictions=predictions)

        return predictions_list, rois, roi_scores, roi_labels


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
            "feature_map_stride": 2,
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
    ).to("cuda:0")

    x = torch.rand((4, 384, 216, 248), device="cuda:0")
    bbox_head(x)


