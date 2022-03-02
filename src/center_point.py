import torch
from torch import nn
from point_pillar_net import PointPillarFeatureNet
from point_pillars_scatter import PointPillarScatter
from point_pillars_backbone import PointPillarBackbone
from center_point_head import CenterBBoxHead
from utils.loss_utils import FocalLossCenterNet, RegLossCenterNet
import utils.centernet_utils as centernet_utils
from config import CenterPointConfig
import numpy as np


class CenterPoint(nn.Module):
    def __init__(self, center_point_config):
        """
        构造函数

        :param center_point_config: 全局配置字典
        """
        super().__init__()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._center_point_config = center_point_config
        self._point_cloud_range = self._center_point_config["POINT_CLOUD_RANGE"]
        self._voxel_size = self._center_point_config["VOXEL_SIZE"]

        # point pillar特征提取网络
        self._point_pillars_feature_net = PointPillarFeatureNet(
            in_channels=4,
            feat_channels=64,
            voxel_size=self._center_point_config["VOXEL_SIZE"],
            point_cloud_range=self._center_point_config["POINT_CLOUD_RANGE"],
        )

        # 伪图像生成器
        self._point_pillars_scatter = PointPillarScatter(
            in_channels=64,
            nx=(self._point_cloud_range[3] - self._point_cloud_range[0]) / self._voxel_size[0],
            ny=(self._point_cloud_range[4] - self._point_cloud_range[1]) / self._voxel_size[1]
        )

        # Backbone特征提取器
        self._point_pillars_backbone = PointPillarBackbone()

        # Center Point检测头
        self._center_point_bbox_head = CenterBBoxHead(
            input_channels=self._center_point_config["HEAD_INPUT_CHANNELS"],
            point_cloud_range=self._center_point_config["POINT_CLOUD_RANGE"],
            voxel_size=self._center_point_config["VOXEL_SIZE"],
            class_names=self._center_point_config["CLASS_NAMES"],
            head_config=self._center_point_config["HEAD_CONFIG"],
            target_assigner_config=self._center_point_config["TARGET_ASSIGNER_CONFIG"],
            post_processing_config=self._center_point_config["POST_PROCESSING_CONFIG"]
        )

        # 构建损失函数
        self.build_losses()

        # 保存每一次 forward 结果
        self._forward_ret_dict = dict()

    def build_losses(self):
        self.add_module("hm_loss_func", FocalLossCenterNet())
        self.add_module("reg_loss_func", RegLossCenterNet())

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
        # heatmap, ret_boxes, inds, masks的初始化，形状分别为 (num_classes, 216, 248), (500, 8), (500,), (500,)
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[0], feature_map_size[1])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1]))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        # gt_boxes的形状为(N, 8), 8代表着 x, y, z, dx, dy, dz, orientation, category_id_in_separate_single_head
        # 其中 dx, dy, dz 也可表示为物体的 length, width, height
        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]

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

            # 根据给定的中心点(center)以及高斯半径(radius)绘制 center-ness score 的热力图，该图会作为训练网络时的拟合目标值
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cls_id_in_curr_separate_head], center[k], radius[k].item())

            # 参考docs/scatter.png中的索引定义
            inds[k] = center_int[k, 0] * feature_map_size[1] + center_int[k, 1]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=(216, 248)):
        """
        针对给定的 ground truth 3D boxes 生成神经网络的回归的目标值

        :param gt_boxes: 类型是一个list数组，因为同一个batch中不同样本的3D boxes真值数量不同，这里以数组形式存储
        :param feature_map_size: 送进 detection head 的特征图大小，如针对KITTI默认配置，使用PointPillar提取特征的话，其尺度
                                 为 (216, 248)
        :return:
        """

        ret_dict = {
            "heatmaps": list(),
            "target_boxes": list(),
            "inds": list(),
            "masks": list(),
            "heatmap_masks": list()
        }

        all_names = np.array(["bg", *self._center_point_config["CLASS_NAMES"]])
        batch_size = len(gt_boxes)

        for index, single_head_class_names in enumerate(self._center_point_bbox_head.class_names_each_head):
            target_hm_list, target_boxes_list, target_inds_list, target_masks_list = list(), list(), list(), list()

            for batch_index in range(batch_size):
                gt_boxes_in_curr_sample = gt_boxes[batch_index]
                gt_class_names_in_curr_sample = all_names[gt_boxes_in_curr_sample[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = list()

                for box_index, name in enumerate(gt_class_names_in_curr_sample):
                    if name not in single_head_class_names:
                        continue

                    box = gt_boxes_in_curr_sample[box_index]
                    box[-1] = single_head_class_names.index(name) + 1
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
                    feature_map_stride=self._center_point_config["TARGET_ASSIGNER_CONFIG"]["FEATURE_MAP_STRIDE"],
                    num_max_objs=self._center_point_config["TARGET_ASSIGNER_CONFIG"]["NUM_MAX_OBJS"],
                    gaussian_overlap=self._center_point_config["TARGET_ASSIGNER_CONFIG"]["GAUSSIAN_OVERLAP"],
                    min_radius=self._center_point_config["TARGET_ASSIGNER_CONFIG"]["MIN_RADIUS"],
                )

                target_hm_list.append(heatmap.to(self._device))
                target_boxes_list.append(boxes.to(self._device))
                target_inds_list.append(inds.to(self._device))
                target_masks_list.append(mask.to(self._device))

            ret_dict['heatmaps'].append(torch.stack(target_hm_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(target_inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(target_masks_list, dim=0))

        # 将当前batch的真值存起来，计算loss时会依赖它
        self._forward_ret_dict["target_dicts"] = ret_dict
        return ret_dict

    @staticmethod
    def sigmoid(x):
        eps = 1e-4
        return torch.clamp(x.sigmoid(), min=eps, max=1.0 - eps)

    def get_loss(self):
        pred_dicts = self._forward_ret_dict['pred_dicts']
        target_dicts = self._forward_ret_dict['target_dicts']

        tb_dict = dict()
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])

            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self._center_point_config["LOSS_CONFIG"]["LOSS_WEIGHTS"]['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([
                pred_dict[head_name] for head_name in
                self._center_point_config["HEAD_CONFIG"]["SEPARATE_HEAD_CONFIG"]["HEAD_ORDER"]], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self._center_point_config["LOSS_CONFIG"]["LOSS_WEIGHTS"]['code_weights'])).sum()
            loc_loss = loc_loss * self._center_point_config["LOSS_CONFIG"]["LOSS_WEIGHTS"]['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def forward(self, voxels, indices, nums_per_voxel, sample_indices):
        """
        前向处理函数

        :param voxels: torch.Tensor, 形状为 (M, max_points_per_voxel, 4). 其中 M 为当前 batch 中点云样本进行体素化后的总体素
                       数量，例如 batch_size = 4时，四个样本的体素化后体素数量分别为 m1, m2, m3, m4. 则 M = m1 + m2 + m3 + m4.
                       4为原始点云的输入维度，分别表示 [x, y, z, intensity]
        :param indices: torch.Tensor, 形状为 (M, 3), 表示每一个体素（Pillar）对应的坐标索引，顺序为 z, y, x
        :param nums_per_voxel: torch.Tensor, 形状为(M, )，表示每一个体素内的有效点云点数
        :param sample_indices: torch.Tensor， 形状为(M, ), 表示当前体素属于 batch 中的哪一个，即索引
        :return:
        """
        # 步骤一：提取体素中的点云特征
        # 输出 pillar_features 的形状为 (M, 64), M 为该 batch 中所有样本进行体素化之后的体素数量总和，
        # 64 为体素中点云进行特征提取后的特征维度
        pillar_features = self._point_pillars_feature_net(voxels, indices, nums_per_voxel, sample_indices)

        # 步骤二：将学习得到的体素点云特征重新转化为伪图像形式
        # 输出 batch_canvas 的维度信息为 (batch_size, C, nx, ny), 论文中 C=64, nx=432, ny=496
        batch_size = int(sample_indices[-1].item() + 1)
        batch_canvas = self._point_pillars_scatter(
            batch_pillar_features=pillar_features,
            batch_indices=indices,
            sample_indices=sample_indices,
            batch_size=batch_size)

        # 步骤三：利用Backbone提取伪图像的特征，输出维度为 (batch_size, 6C, nx/2, ny/2)
        backbone_feats = self._point_pillars_backbone(batch_canvas=batch_canvas)

        # 步骤五：基于CenterPoint Head对3D物体进行目标检测和回归
        predictions_list, rois, roi_scores, roi_labels = self._center_point_bbox_head(x=backbone_feats)

        # 将预测结果存储起来，计算loss时会依赖它
        self._forward_ret_dict["pred_dicts"] = predictions_list

        return predictions_list, rois, roi_scores, roi_labels


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    center_point_model = CenterPoint(center_point_config=CenterPointConfig).to(device)

    from center_point_dataset import CenterPointDataset
    from torch.utils.data import DataLoader

    train_dataset = CenterPointDataset(
        voxel_size=CenterPointConfig["VOXEL_SIZE"],
        class_names=CenterPointConfig["CLASS_NAMES"],
        point_cloud_range=CenterPointConfig["POINT_CLOUD_RANGE"],
        phase="train",
        dataset_config=CenterPointConfig["DATASET_CONFIG"]
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=CenterPointConfig["TRAIN_CONFIG"]["BATCH_SIZE"],
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

        batch_voxels = batch_voxels.to(device)
        batch_indices = batch_indices.to(device)
        batch_nums_per_voxel = batch_nums_per_voxel.to(device)
        batch_sample_indices = batch_sample_indices.to(device)

        print("\n")

        center_point_model(
            voxels=batch_voxels,
            indices=batch_indices,
            nums_per_voxel=batch_nums_per_voxel,
            sample_indices=batch_sample_indices)

        break
