import torch
import src.utils.centernet_utils as centernet_utils
import numpy as np
import matplotlib.pyplot as plt
from gt_encode_demo import assign_target_of_single_head


def assign_targets(gt_boxes, feature_map_size=None):
    ret_dict = {
        "heatmaps": list(),
        "target_boxes": list(),
        "inds": list(),
        "masks": list(),
        "heatmap_masks": list()
    }

    all_names = np.array(["bg", 'Vehicle', 'Pedestrian'])
    batch_size = len(gt_boxes)
    print("batch_size = {}".format(batch_size))

    for index, single_head_class_names in enumerate([['Vehicle', 'Pedestrian']]):
        target_hm_list, target_boxes_list, target_inds_list, target_masks_list = list(), list(), list(), list()

        for batch_index in range(batch_size):
            gt_boxes_in_curr_sample = gt_boxes[batch_index]
            print(gt_boxes_in_curr_sample)
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

            heatmap, boxes, inds, mask = assign_target_of_single_head(
                point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                voxel_size=[0.16, 0.16, 4],
                num_classes=len(single_head_class_names),
                gt_boxes=gt_boxes_single_head.cpu(),
                feature_map_size=feature_map_size,
                feature_map_stride=2,
                num_max_objs=500,
                gaussian_overlap=0.1,
                min_radius=2,
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


if __name__ == "__main__":
    gt_boxes = torch.from_numpy(np.array([[
        [62, 36.0, 1.0, 5.3, 1.7, 1.5, 0.0, 1],  # 最后一列为类别信息 (+1 处理过)，指的是全局类别id, 而不是separate head中的类别id
        [30, 15, 1.0, 0.4, 0.3, 1.9, 0.72, 2],  # 最后一列为类别信息 (+1 处理过)，指的是全局类别id, 而不是separate head中的类别id
    ]]))

    ret_dict = assign_targets(gt_boxes=gt_boxes, feature_map_size=[216, 248])

    heatmaps = ret_dict["heatmaps"][0]
    target_boxes = ret_dict["target_boxes"][0]
    inds = ret_dict["inds"][0]
    masks = ret_dict["masks"][0]

    print("heatmaps.shape = {}".format(heatmaps.shape))
    print("target_boxes.shape = {}".format(target_boxes.shape))
    print("inds.shape = {}".format(inds.shape))
    print("masks.shape = {}".format(masks.shape))

    valid_indices = inds[masks > 0]
    print(valid_indices)
    print(target_boxes[0][0])
    print(target_boxes[0][1])
    print(target_boxes[0][2])

    xs = torch.div(valid_indices, 248, rounding_mode='floor').float()
    ys = (valid_indices % 248).int().float()
    print(xs, ys)

    batch_hm = heatmaps.to("cuda:0")
    batch_gt_npy = torch.zeros(size=(1, 216, 248, 8))
    batch_gt_npy[0][193][236][:] = target_boxes[0][0]
    batch_gt_npy[0][93][170][:] = target_boxes[0][1]
    batch_gt_npy = batch_gt_npy.permute(0, 3, 1, 2).contiguous()

    batch_rot_cos = torch.unsqueeze(batch_gt_npy[:, 6, :, :], dim=1).to("cuda:0")
    batch_rot_sin = torch.unsqueeze(batch_gt_npy[:, 7, :, :], dim=1).to("cuda:0")
    batch_center = batch_gt_npy[:, 0:2, :, :].to("cuda:0")
    batch_center_z = torch.unsqueeze(batch_gt_npy[:, 3, :, :], dim=1).to("cuda:0")
    batch_dim = batch_gt_npy[:, 3:6, :, :].to("cuda:0")

    # center.shape = torch.Size([4, 2, 216, 248])
    # center_z.shape = torch.Size([4, 1, 216, 248])
    # dim.shape = torch.Size([4, 3, 216, 248])
    # rot_sin.shape = torch.Size([4, 1, 216, 248])
    # rot_cos.shape = torch.Size([4, 1, 216, 248])



    print("================ decode ====================")
    decoded_predictions = centernet_utils.decode_bbox_from_heatmap(
        heatmap=batch_hm,
        rot_cos=batch_rot_cos,
        rot_sin=batch_rot_sin,
        center=batch_center,
        center_z=batch_center_z,
        dim=batch_dim,
        vel=None,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=[0.16, 0.16, 4.0],
        feature_map_stride=2,
        K=500,
        circle_nms=False,
        score_thresh=0.10,
        post_center_limit_range=torch.tensor([0, -39.68, -3, 69.12, 39.68, 1]).cuda().float()
    )

    print("decoded_predictions = {}".format(decoded_predictions))


    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(heatmaps[0][0].numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(heatmaps[0][1].numpy())
    plt.show()
