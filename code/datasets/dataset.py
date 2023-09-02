from PIL import Image
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False, fisheye=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        self.fisheye = fisheye
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.fisheye:
            mask_path = self.data.iloc[idx, 2]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask[mask == 255] = 12 #배경을 픽셀값 12로 간주
            height, width = image.shape[:2]

            focal_length = width / 4
            center_x = width / 2
            center_y = height / 2
            camera_matrix = np.array([[focal_length, 0, center_x],
                              [0, focal_length, center_y],
                              [0, 0, 1]], dtype=np.float32)

            dist_coeffs = np.array([0, 0.5, 0, 0], dtype=np.float32)
            undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

            
            if self.infer:
                if self.transform:
                    image = self.transform(image=undistorted_image)['image']
                return image
            
            if self.transform:
                augmented = self.transform(image=undistorted_image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                
        else:
            if self.infer:
                if self.transform:
                    image = self.transform(image=image)['image']
                return image
            
            mask_path = self.data.iloc[idx, 2]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask[mask == 255] = 12 #배경을 픽셀값 12로 간주

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
        
        

        return image, mask