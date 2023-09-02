import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets.dataset import *
from models.model import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    transform = A.Compose(
        [   
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    train_dataset = CustomDataset(csv_file='./data/train_source.csv', transform=transform, fisheye=True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    val_dataset = CustomDataset(csv_file='./data/val_source.csv', transform=transform, fisheye=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)
    
        # model 초기화
    model = UNet().to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    # loss function과 optimizer 정의
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = 99999.0

    # training loop
    for epoch in range(50):  # 20 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_dataloader):
            images = images.float().to(device)
            masks = masks.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f'Train Epoch {epoch+1}, Loss: {epoch_loss/len(train_dataloader)}')
        
        model.eval()
        print("======================")
        print("Validation starting...")
        print("======================")
        
        val_losses = 0.0
        with torch.no_grad():
            for images, masks in val_dataloader:   
                images = images.float().to(device)
                masks = masks.long().to(device)
                outputs = model(images)
                val_loss = criterion(outputs, masks.squeeze(1))
                val_losses += val_loss.item()
                
            val_epoch_loss = val_losses/len(val_dataloader)
        
        print(f'Val Epoch {epoch+1}, Loss: {val_epoch_loss}')
        if best_loss > val_epoch_loss:
            best_loss = val_epoch_loss
            print('Updated best model, Saving weights...')
            if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), './weights' + f'/best_model_{val_losses / len(val_dataloader):.5f}_{epoch}.pth')
            else:
                torch.save(model.state_dict(), './weights' + f'/best_model_{val_losses / len(val_dataloader):.5f}_{epoch}.pth')      
        
         
    

if __name__ == '__main__':
    main()