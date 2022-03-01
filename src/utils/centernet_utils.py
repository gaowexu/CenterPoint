import torch
import torch.nn.functional as F
import numpy as np
import numba


def gaussian_radius(height, width, min_overlap=0.5):
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    ret = torch.min(torch.min(r1, r2), r3)
    return ret


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1, valid_mask=None):
    """
    根据给定的中心点(center)以及高斯半径(radius)绘制 center-ness score 的热力图，该图会作为训练网络时的
    拟合目标值

    :param heatmap: torch.tensor, 形状为 torch.Size([216, 248])
    :param center: torch.tensor, (x, y)
    :param radius: float
    :param k:
    :param valid_mask:
    :return:
    """
    # 直径
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    # height 指的是 x 方向，即车辆前进方向 （参考KITTI Lidar 坐标系）
    # width 指的是 y 方向 （参考KITTI Lidar 坐标系）
    height, width = heatmap.shape[0:2]

    left = min(y, radius)
    right = min(width - y, radius + 1)
    top = min(x, radius)
    bottom = min(height - x, radius + 1)

    masked_heatmap = heatmap[x-top:x+bottom, y-left: y+right]

    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]).to(heatmap.device).float()

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        if valid_mask is not None:
            cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
            masked_gaussian = masked_gaussian * cur_valid_mask.float()

        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


@numba.jit(nopython=True)
def circle_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]
    keep = torch.from_numpy(keep).long().to(boxes.device)
    return keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _top_k(scores, K=40):
    """

    :param scores: torch.tensor, 形状为 (batch_size, num_classes, 216, 248)
    :param K:
    :return:
    """
    batch_size, num_class, height, width = scores.size()

    # 展平最后两个维度，scores_flatten.shape = torch.Size([1, 2, 53568]), 其中 55368 = 216*248
    scores_flatten = scores.flatten(2, 3)

    # 首先针对每一个类别的 score map, 挑选前 K 个score保存，并保存其索引信息
    # topk_scores.shape = torch.Size([batch_size, num_classes, K])
    # topk_inds.shape = torch.Size([batch_size, num_classes, K])
    topk_scores, topk_inds = torch.topk(scores_flatten, K)

    # 根据全局索引得到 x, y 方向的索引
    # topk_xs.shape = torch.Size([batch_size, num_classes, K])
    # topk_ys.shape = torch.Size([batch_size, num_classes, K])
    topk_inds = topk_inds % (height * width)
    topk_xs = torch.div(topk_inds, width, rounding_mode='floor').float()
    topk_ys = (topk_inds % width).int().float()

    # 将topk_scores展平，展平后形状为 torch.Size([batch_size, num_classes * K])
    topk_scores_flatten = topk_scores.view(batch_size, -1)

    # 选出所有类别中 top K 的 score, 并将其索引保存下来
    topk_score, topk_ind = torch.topk(topk_scores_flatten, K)

    # 计算得到所属的类别id, topk_classes输出形状为 (batch_size, K)
    topk_classes = torch.div(topk_ind, K, rounding_mode='floor').int()

    # 计算得到 top K的 scores对应的全局索引， 形状为 (batch_size, K)
    topk_inds = _gather_feat(topk_inds.view(batch_size, -1, 1), topk_ind).view(batch_size, K)

    # 计算得到 top K的 scores对应的 x,y 索引， 形状为 (batch_size, K)
    topk_xs = _gather_feat(topk_xs.view(batch_size, -1, 1), topk_ind).view(batch_size, K)
    topk_ys = _gather_feat(topk_ys.view(batch_size, -1, 1), topk_ind).view(batch_size, K)

    return topk_score, topk_inds, topk_classes, topk_xs, topk_ys


def decode_bbox_from_heatmap(heatmap, rot_cos, rot_sin, center, center_z, dim,
                             point_cloud_range=None, voxel_size=None, feature_map_stride=None,
                             vel=None, K=100, circle_nms=False, score_thresh=None,
                             post_center_limit_range=None):
    """
    从预测得到的dense feature map中解码出 3D bounding boxes信息

    :param heatmap:
    :param rot_cos:
    :param rot_sin:
    :param center:
    :param center_z:
    :param dim:
    :param point_cloud_range:
    :param voxel_size:
    :param feature_map_stride:
    :param vel:
    :param K:
    :param circle_nms:
    :param score_thresh:
    :param post_center_limit_range:
    :return:
    """

    batch_size, num_class, _, _ = heatmap.size()

    if circle_nms:
        # TODO: not checked yet
        assert False, 'not checked yet'
        heatmap = _nms(heatmap)

    scores, inds, class_ids, xs, ys = _top_k(heatmap, K=K)

    print("center.shape = {}".format(center.shape))
    print("center_z.shape = {}".format(center_z.shape))
    print("dim.shape = {}".format(dim.shape))
    print("rot_sin.shape = {}".format(rot_sin.shape))
    print("rot_cos.shape = {}".format(rot_cos.shape))

    center = _transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
    center_z = _transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)
    rot_sin = _transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
    rot_cos = _transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)

    angle = torch.atan2(rot_sin, rot_cos)
    xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]
    ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]

    xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

    box_part_list = [xs, ys, center_z, dim, angle]
    if vel is not None:
        vel = _transpose_and_gather_feat(vel, inds).view(batch_size, K, 2)
        box_part_list.append(vel)

    final_box_preds = torch.cat((box_part_list), dim=-1)
    final_scores = scores.view(batch_size, K)
    final_class_ids = class_ids.view(batch_size, K)

    assert post_center_limit_range is not None
    mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
    mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)

    if score_thresh is not None:
        mask &= (final_scores > score_thresh)

    ret_pred_dicts = []
    for k in range(batch_size):
        cur_mask = mask[k]
        cur_boxes = final_box_preds[k, cur_mask]
        cur_scores = final_scores[k, cur_mask]
        cur_labels = final_class_ids[k, cur_mask]

        if circle_nms:
            assert False, 'not checked yet'
            centers = cur_boxes[:, [0, 1]]
            boxes = torch.cat((centers, scores.view(-1, 1)), dim=1)
            keep = _circle_nms(boxes, min_radius=min_radius, post_max_size=nms_post_max_size)

            cur_boxes = cur_boxes[keep]
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]

        ret_pred_dicts.append({
            'pred_boxes': cur_boxes,
            'pred_scores': cur_scores,
            'pred_labels': cur_labels
        })
    return ret_pred_dicts
