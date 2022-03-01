import torch
import numpy as np


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


a = torch.from_numpy(np.array([[
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ],

    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24],
    ],
]]))

b = a.flatten(2, 3)
topk_scores, topk_inds = torch.topk(b, 10)
batch_size, num_class, height, width = a.size()

print(a.shape)
print(b.shape)
print(a)
print("\n")
print(b)
print("\n")
print("topk_scores = {}, shape = {}".format(topk_scores, topk_scores.shape))
print("topk_inds = {}, shape = {}".format(topk_inds, topk_inds.shape))
print("\n")
topk_inds = topk_inds % (height * width)
topk_xs = torch.floor_divide(topk_inds, width).float()
topk_ys = (topk_inds % width).int().float()

print("topk_xs = \n{}".format(topk_xs))
print("topk_ys = \n{}".format(topk_ys))
print("\n")
print("topk_scores.shape = {}".format(topk_scores.shape))
topk_scores_flatten = topk_scores.view(batch_size, -1)
print("topk_scores_flatten.shape = {}".format(topk_scores_flatten.shape))

topk_score, topk_ind = torch.topk(topk_scores_flatten, 10)
print("topk_score.shape = {}, topk_score = {}".format(topk_score.shape, topk_score))
print("topk_ind.shape = {}, topk_ind = {}".format(topk_ind.shape, topk_ind))

topk_classes = torch.floor_divide(topk_ind, 10).int()
print("topk_classes.shape = {}, topk_classes = {}".format(topk_classes.shape, topk_classes))

topk_inds = _gather_feat(topk_inds.view(batch_size, -1, 1), topk_ind).view(batch_size, 10)
topk_ys = _gather_feat(topk_ys.view(batch_size, -1, 1), topk_ind).view(batch_size, 10)
topk_xs = _gather_feat(topk_xs.view(batch_size, -1, 1), topk_ind).view(batch_size, 10)

print("topk_inds.shape = {}, topk_inds = {}".format(topk_inds.shape, topk_inds))
print("topk_xs.shape = {}, topk_xs = {}".format(topk_xs.shape, topk_xs))
print("topk_ys.shape = {}, topk_ys = {}".format(topk_ys.shape, topk_ys))
