import sys
sys.path.append("../")
import torch
from center_point_dataset import CenterPointDataset
from center_point_config import CenterPointConfig
from torch.utils.data import DataLoader
from center_point import CenterPoint
import os
import torch.nn as nn
from functools import partial
import torch.optim as optim
from fastai_optim import OptimWrapper
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('../runs/center_point')


def build_optimizer(model, optim_cfg):
    def children(m: nn.Module):
        return list(m.children())

    def num_children(m: nn.Module) -> int:
        return len(children(m))

    flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
    get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

    optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
    optimizer = OptimWrapper.create(
        optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg["WEIGHT_DECAY"], true_wd=True, bn_wd=True
    )

    return optimizer


def train_one_epoch(epoch_index, dataloader, model, optimizer, device):
    model.train()
    for batch_index, (batch_voxels, batch_indices, batch_nums_per_voxel,
                      batch_sample_indices, batch_gt_3d_boxes_list) in enumerate(dataloader):
        batch_voxels = batch_voxels.to(device)
        batch_indices = batch_indices.to(device)
        batch_nums_per_voxel = batch_nums_per_voxel.to(device)
        batch_sample_indices = batch_sample_indices.to(device)

        # forward
        predictions_list, rois, roi_scores, roi_labels = model(
            voxels=batch_voxels,
            indices=batch_indices,
            nums_per_voxel=batch_nums_per_voxel,
            sample_indices=batch_sample_indices
        )

        # assign targets
        # model.assign_targets(gt_boxes=batch_gt_3d_boxes_list, feature_map_size=[216, 248]) # for KITTI dataset
        model.assign_targets(gt_boxes=batch_gt_3d_boxes_list, feature_map_size=[248, 496])  # for carla dataset

        # compute loss
        loss, tb_dict = model.get_loss()

        # loss back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar(
            "training_loss",
            loss,
            epoch_index * CenterPointConfig["TRAIN_CONFIG"]["BATCH_SIZE"] + batch_index)

        print("Batch = {}/{}: loss = {:.4f}, "
              "hm_loss_head_0 = {:.4f}, loc_loss_head_0 = {:.4f}, "
              "rois.shape = {}".format(
            batch_index+1, len(dataloader), loss,
            tb_dict["hm_loss_head_0"], tb_dict["loc_loss_head_0"],
            rois.shape
        ))


def main():
    # Step 1: define dataset
    train_dataset = CenterPointDataset(
        voxel_size=CenterPointConfig["VOXEL_SIZE"], class_names=CenterPointConfig["CLASS_NAMES"],
        point_cloud_range=CenterPointConfig["POINT_CLOUD_RANGE"], phase="train",
        dataset_config=CenterPointConfig["DATASET_CONFIG"],
        augmentation_config=CenterPointConfig["DATA_AUGMENTATION_CONFIG"],
        dataset_info_config=CenterPointConfig["DATASET_CONFIG"])

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=CenterPointConfig["TRAIN_CONFIG"]["BATCH_SIZE"], sampler=None, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_batch)

    val_dataset = CenterPointDataset(
        voxel_size=CenterPointConfig["VOXEL_SIZE"], class_names=CenterPointConfig["CLASS_NAMES"],
        point_cloud_range=CenterPointConfig["POINT_CLOUD_RANGE"], phase="val",
        dataset_config=CenterPointConfig["DATASET_CONFIG"],
        augmentation_config=CenterPointConfig["DATA_AUGMENTATION_CONFIG"],
        dataset_info_config=CenterPointConfig["DATASET_CONFIG"])

    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=CenterPointConfig["TRAIN_CONFIG"]["BATCH_SIZE"], sampler=None, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_batch)

    print("There are totally {} samples in training dataset.".format(len(train_dataset)))
    print("There are totally {} samples in test dataset.".format(len(val_dataset)))
    print("Training Batches = {}".format(len(train_dataloader)))
    print("Test Batches = {}".format(len(val_dataloader)))

    # Step 2: Define model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if CenterPointConfig["TRAIN_CONFIG"]["PRE_TRAINED_WEIGHTS_PATH"] is None:
        center_point_model = CenterPoint(center_point_config=CenterPointConfig).to(device)
    else:
        print("Loading pre-trained model from {}...".format(
            CenterPointConfig["TRAIN_CONFIG"]["PRE_TRAINED_WEIGHTS_PATH"]))
        center_point_model = torch.load(
            CenterPointConfig["TRAIN_CONFIG"]["PRE_TRAINED_WEIGHTS_PATH"], map_location=device)

    # Step 3: Start training
    # optimizer = torch.optim.Adam(
    #     params=center_point_model.parameters(),
    #     lr=CenterPointConfig["OPTIMIZATION"]["LEARNING_RATE"],
    #     weight_decay=CenterPointConfig["OPTIMIZATION"]["WEIGHT_DECAY"]
    # )

    optimizer = build_optimizer(model=center_point_model, optim_cfg=CenterPointConfig["OPTIMIZATION"])

    max_epochs = CenterPointConfig["TRAIN_CONFIG"]["MAX_EPOCHS"]
    for epoch_index in range(max_epochs):
        print("------------------------------- Epoch {} ----------------------------------".format(epoch_index + 1))
        # ????????????epoch
        train_one_epoch(
            epoch_index=epoch_index,
            dataloader=train_dataloader,
            model=center_point_model,
            optimizer=optimizer,
            device=device)

        # ????????????
        if epoch_index % 10 == 0:
            model_save_full_path = os.path.join(
                CenterPointConfig["MODEL_SAVE_ROOT_DIR"],
                "center_point_epoch_{}.pth".format(epoch_index))
            torch.save(center_point_model, model_save_full_path)

        # # ???????????????????????????
        # evaluate(
        #     dataloader=val_dataloader,
        #     model=center_point_model,
        #     loss_fn=loss_handler,
        #     device=device)


if __name__ == "__main__":
    main()
