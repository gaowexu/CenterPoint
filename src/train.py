import torch
from center_point_dataset import CenterPointDataset
from config import CenterPointConfig
from torch.utils.data import DataLoader
from center_point import CenterPoint


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    samples_amount = len(dataloader.dataset)
    model.train()
    for batch_index, (batch_voxels, batch_indices, batch_nums_per_voxel,
                      batch_sample_indices, batch_gt_3d_boxes_list) in enumerate(dataloader):
        input_image_batch = input_image_batch.to(device)
        input_gt_batch = input_gt_batch.to(device)

        # compute loss
        logits, probs = model(input_image_batch)
        loss = loss_fn(logits, input_gt_batch)

        # loss back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 10 == 0:
            loss, current = loss.item(), batch_index * len(input_image_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{samples_amount:>5d}]")


def main():
    # Step 1: define dataset
    train_dataset = CenterPointDataset(phase="train", dataset_config=CenterPointConfig["dataset_config"])
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=8, sampler=None, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_batch)

    val_dataset = CenterPointDataset(phase="val", dataset_config=CenterPointConfig["dataset_config"])
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=8, sampler=None, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_batch)

    print("There are totally {} samples in training dataset.".format(len(train_dataset)))
    print("There are totally {} samples in test dataset.".format(len(val_dataset)))
    print("Training Batches = {}".format(len(train_dataloader)))
    print("Test Batches = {}".format(len(val_dataloader)))

    # Step 2: Define model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    center_point_model = CenterPoint().to(device)

    # Step 3: Start training
    optimizer = torch.optim.Adam(
        params=center_point_model.parameters(),
        lr=CenterPointConfig["optimization"]["lr"],
        weight_decay=CenterPointConfig["optimization"]["weight_decay"]
    )
    loss_handler = torch.nn.CrossEntropyLoss()

    max_epochs = CenterPointConfig["train_config"]["max_epochs"]
    for epoch_index in range(max_epochs):
        print("------------------------------- Epoch {} ----------------------------------".format(epoch_index + 1))
        # 训练一个epoch
        train_one_epoch(
            dataloader=train_dataloader,
            model=center_point_model,
            loss_fn=loss_handler,
            optimizer=optimizer,
            device=device)

        # # 在验证集上评估性能
        # evaluate(
        #     dataloader=val_dataloader,
        #     model=center_point_model,
        #     loss_fn=loss_handler,
        #     device=device)


if __name__ == "__main__":
    main()
