import os
import argparse

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import satlaspretrain_models

from dataset import FireDataset, split_dataset
from trainer import train, test, run, dice_loss

def main():
    parser = argparse.ArgumentParser(description="Segmentation task on the fire events")
    parser.add_argument('--img_dir', default='dataset-post_fire', type=str)
    parser.add_argument('--mask_dir', default='dataset-masks', type=str)
    parser.add_argument('--num_data', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--model_path', default="test_model.pth", type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument("--multispectral", action="store_true")
    parser.add_argument("--model", choices=["SwinB_MS", "SwinB_RGB"], default="SwinB_MS", type=str)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## Create dataloader
    train_data, val_data, test_data = split_dataset(img_dir=args.img_dir, num_datapoints=args.num_data)

    transform = transforms.Compose([
                transforms.RandomCrop(224, padding=4),
                ])
    train_dataset = FireDataset(img_dir=args.img_dir, mask_dir=args.mask_dir, img_list=train_data, transforms=transform, multispectral=args.multispectral)
    val_dataset = FireDataset(img_dir=args.img_dir, mask_dir=args.mask_dir, img_list=val_data, transforms=transform, multispectral=args.multispectral)
    test_dataset = FireDataset(img_dir=args.img_dir, mask_dir=args.mask_dir, img_list=test_data, transforms=transform, multispectral=args.multispectral)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Initialize weights
    weights_manager = satlaspretrain_models.Weights()
    if args.model == "SwinB_MS":
        # The pretrained model on Sentinel-2, Choose segmentation head for binary classes, Load backbone + FPN
        model = weights_manager.get_pretrained_model(model_identifier="Sentinel2_SwinB_SI_MS",
                                                    fpn=True,
                                                    num_categories=2,
                                                    head=satlaspretrain_models.Head.SEGMENT,
                                                    device=device).to(device)
    elif args.model == "SwinB_RGB":
        model = weights_manager.get_pretrained_model(model_identifier="Sentinel2_SwinB_SI_RGB",
                                                    fpn=True,
                                                    num_categories=2,
                                                    head=satlaspretrain_models.Head.SEGMENT,
                                                    device=device).to(device)
    else:
        raise ValueError("Model is not implemented yet.")
        
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run(model, args.epochs, train_dataloader, val_dataloader, optimizer, dice_loss, scheduler, device)

    torch.save(model.state_dict(), args.model_path)
    print(f"Save model to {args.model_path}")

    _, dice, iou = test(model=model, test_dataloader=test_dataloader, criterion=dice_loss, device=device)
    print(f"Evaluation: dice score {dice}, iou {iou}")


if __name__ == '__main__':
    main()

