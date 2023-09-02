from glob import glob
import os
import cv2
import albumentations as A

from torch.utils.data import DataLoader, Dataset

class CIDATrainDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.imgList = glob(os.path.join(dir, 'train_source_image'))
        self.GTList = glob(os.path.join(dir, 'train_source_gt'))
        
        self.transform = transform
        
    def __len__(self):
        return len(self.imgList)
    
    def __getitem__(self, index):
        img = cv2.imread(self.imgList[index])
        gt = cv2.imread(self.GTList[index])
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        
        transformed = self.transform(image=img, mask = gt)
        img = transformed['image']
        gt = transformed['mask']
        
        return dict(image=img, gt=gt)
    
transform = A.Compose()