import torch
from center_point_dataset import CenterPointDataset
from config import CenterPointConfig
from torch.utils.data import DataLoader
from center_point import CenterPoint


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    samples_amount = len(dataloader.dataset)
    model.train()
    for batch_idx, (input_image_batch, input_gt_batch) in enumerate(dataloader):
        input_image_batch = input_image_batch.to(device)
        input_gt_batch = input_gt_batch.to(device)

        # compute loss
        logits, probs = model(input_image_batch)
        loss = loss_fn(logits, input_gt_batch)

        # loss back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            loss, current = loss.item(), batch_idx * len(input_image_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{samples_amount:>5d}]")


def evaluate(dataloader, model, loss_fn, device):
    samples_amount = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for input_image_batch, input_gt_batch in dataloader:
            input_image_batch = input_image_batch.to(device)
            input_gt_batch = input_gt_batch.to(device)

            logits, probs = model(input_image_batch)
            test_loss += loss_fn(logits, input_gt_batch).item()

            correct += (probs.argmax(1) == input_gt_batch).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= samples_amount
    print(f"Test Performance: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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
    adam_optimizer = torch.optim.Adam(custom_model.parameters(), lr=1e-4)
    loss_handler = torch.nn.CrossEntropyLoss()

    max_epochs = CenterPointConfig["train_config"]["max_epochs"]
    for epoch_index in range(max_epochs):
        print("------------------------------- Epoch {} ----------------------------------".format(epoch_index + 1))
        # 训练一个epoch
        train_one_epoch(
            dataloader=train_dataloader,
            model=center_point_model,
            loss_fn=loss_handler,
            optimizer=adam_optimizer,
            device=device)

        # 在验证集上评估性能
        evaluate(
            dataloader=val_dataloader,
            model=center_point_model,
            loss_fn=loss_handler,
            device=device)


if __name__ == "__main__":
    main()
