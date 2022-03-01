import torch
import utils.centernet_utils as centernet_utils


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
    x, y, z, dx, dy, dz = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]

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


if __name__ == "__main__":
    import numpy as np
    gt_boxes = torch.from_numpy(np.array([
        [50.0, 0, 1.0, 5.3, 1.7, 1.5, 0.0, 1],
        [10, 15, 1.0, 0.4, 0.3, 1.9, 0.72, 2],
    ]))

    heatmap, ret_boxes, inds, mask = assign_target_of_single_head(
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=[0.16, 0.16, 4.0],
        num_classes=2,
        gt_boxes=gt_boxes,
        feature_map_size=[216, 248],
        feature_map_stride=2,
    )

    print("heatmap.shape = {}".format(heatmap.shape))
    print("ret_boxes.shape = {}".format(ret_boxes.shape))
    print("inds.shape = {}".format(inds.shape))
    print("mask.shape = {}".format(mask.shape))


    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(heatmap[0].numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap[1].numpy())
    plt.show()


