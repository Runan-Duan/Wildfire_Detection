import torch
import numpy as np
import torch.nn.functional as F

def dice_loss(outputs, targets):
    num_classes=2
    probs = F.softmax(outputs[0], dim=1)  # Compute prob across number of classes

    if probs.shape[2:] != targets.shape[1:]:
        probs = F.interpolate(probs, size=targets.shape[1:], mode='bilinear', align_corners=False)

    # One-hot encoding with tensor shape (N, C, H, W)
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float().to(targets.device)


    dims = (0, 2, 3)
    epsilon = 1e-6
    # Compute dice for fire area and background
    intersection = torch.sum(probs * targets_one_hot, dim=dims)
    cardinality = torch.sum(probs + targets_one_hot, dim=dims)
    union = cardinality - intersection
    dice_per_class = (2. * intersection + epsilon) / (cardinality + epsilon)
    iou_per_class = (intersection + epsilon) / (union + epsilon)

    # Handle imbalanced classes
    class_counts = torch.sum(targets_one_hot, dim=dims)
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights = class_weights / class_weights.sum()
    weights = torch.tensor(class_weights, device=targets.device)

    # Weighted dice
    weighted_dice = torch.sum(dice_per_class * weights)
    weighted_iou = torch.sum(iou_per_class * weights) / weights.sum()

    log_cosh_loss = torch.log(torch.cosh(1.0 - weighted_dice))

    return log_cosh_loss, weighted_dice, weighted_iou


def train(model, train_dataloader, optimizer, criterion, device):
    model.train()
    sum_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, _, _ = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        num_samples = inputs.size(0)
        sum_loss += loss.item() * num_samples
  
    average_loss = sum_loss / len(train_dataloader.dataset)
    return average_loss # avg epoch loss

def test(model, test_dataloader, criterion, device):
    model.eval()
    sum_loss = 0
    sum_dice = 0
    sum_iou = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss, dice_score, iou = criterion(outputs, targets)
            num_samples = inputs.size(0)
            sum_loss += loss.item() * num_samples
            sum_dice += dice_score.item() * num_samples
            sum_iou += iou.item() * num_samples
          
    average_loss = sum_loss / len(test_dataloader.dataset)
    average_dice = sum_dice / len(test_dataloader.dataset)
    average_iou = sum_iou / len(test_dataloader.dataset)
    return average_loss, average_dice, average_iou


def run(model, epochs, train_dataloader, val_dataloader, optimizer, criterion, scheduler, device):
    for epoch in range(1, epochs + 1):
        train_loss = train(model,
                           train_dataloader=train_dataloader,
                           optimizer=optimizer,
                           criterion=criterion,
                           device=device,)
        val_loss, val_dice, val_iou = test(model,
                            test_dataloader=val_dataloader,
                            criterion=criterion,
                            device=device,)
        scheduler.step()
        print(f"Epoch [{epoch}/{epochs}]: train loss {train_loss} | val loss {val_loss} | val dice {val_dice} | val iou {val_iou}")

