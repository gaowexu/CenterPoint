import torch
import src.utils.centernet_utils as centernet_utils
import numpy as np
import matplotlib.pyplot as plt
from gt_encode_demo import assign_target_of_single_head


if __name__ == "__main__":
    gt_boxes = torch.from_numpy(np.array([
        [62, 36.0, 1.0, 5.3, 1.7, 1.5, 0.0, 1],  # 最后一列为类别信息 (+1 处理过)，指的是全局类别id, 而不是separate head中的类别id
        [30, 15, 1.0, 0.4, 0.3, 1.9, 0.72, 2],  # 最后一列为类别信息 (+1 处理过)，指的是全局类别id, 而不是separate head中的类别id
    ]))

    heatmap, ret_boxes, inds, mask = assign_target_of_single_head(
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=[0.16, 0.16, 4.0],
        num_classes=2,
        gt_boxes=gt_boxes,
        feature_map_size=[216, 248],
        feature_map_stride=2,
    )

    print("ret_boxes.shape = {}".format(ret_boxes.shape))

    batch_hm = heatmap[None, :]
    ret_boxes = ret_boxes[None, :]
    print(ret_boxes.shape)
    batch_rot_cos = ret_boxes[:, :, 6]
    batch_rot_sin = ret_boxes[:, :, 7]
    batch_center = ret_boxes[:, :, 0:2]
    batch_center_z = ret_boxes[:, :, 2]
    batch_dim = ret_boxes[:, :, 3:6]

    # torch.Size([4, 2, 216, 248])
    # torch.Size([4, 2, 216, 248])
    # torch.Size([4, 1, 216, 248])
    # torch.Size([4, 3, 216, 248])
    # torch.Size([4, 1, 216, 248])
    # torch.Size([4, 1, 216, 248])

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(heatmap[0].numpy())
    # plt.subplot(1, 2, 2)
    # plt.imshow(heatmap[1].numpy())
    # plt.show()


    # decoded_predictions = centernet_utils.decode_bbox_from_heatmap(
    #     heatmap=batch_hm,
    #     rot_cos=batch_rot_cos,
    #     rot_sin=batch_rot_sin,
    #     center=batch_center,
    #     center_z=batch_center_z,
    #     dim=batch_dim,
    #     vel=None,
    #     point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
    #     voxel_size=[0.16, 0.16, 4.0],
    #     feature_map_stride=2,
    #     K=500,
    #     circle_nms=False,
    #     score_thresh=0.10,
    #     post_center_limit_range=torch.tensor([0, -39.68, -3, 69.12, 39.68, 1]).cuda().float()
    # )





