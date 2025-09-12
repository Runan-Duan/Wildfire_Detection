import os
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage
import rasterio as rio
import torch
import random
import torchvision.transforms as transforms

### ==============================================================================================
### Split train, validation and test samples
### These samples are raw satellite imagery from the Google Earth
### ==============================================================================================
def split_dataset(img_dir, train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1, seed = 0, num_datapoints = 100):
    data = os.listdir(img_dir)
    if num_datapoints <= len(data):
        data_size = num_datapoints
    else:
        data_size = len(data)

    train_size = int(data_size * train_ratio)
    val_size = int(data_size * val_ratio)
    test_size = int(data_size * test_ratio)
    print(f"Training data size {train_size}, Validation data size {val_size}, Testing data size {test_size}")

    # Shuffle all images and split into three subsets
    random.seed(seed)
    random.shuffle(data)
    train_data = data[:train_size]
    val_data = data[train_size : train_size+val_size]
    test_data = data[train_size+val_size : data_size]

    return train_data, val_data, test_data

### ==============================================================================================
### Process raw images by filling in NaNs, normalizing values, cropping images, and mapping channels
### ==============================================================================================
class FireDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_list, transforms, multispectral=False):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir   # 0, 1 unburned or burned
        self.img_list = img_list
        self.transforms = transforms
        self.multispectral = multispectral

    def __len__(self):
        return len(self.img_list)

    def fill_nan(self, data):
        
        filled_data = np.zeros_like(data)

        for idx in range(data.shape[0]):
            band = data[idx,:, :].copy()   # 2d
            
            mask = np.isnan(band)  # True is nan
            if np.any(mask):
                # Replace nan with nearest non-nan values
                indices = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
                band = band[tuple(indices)]
            filled_data[idx] = band  

        return filled_data

    def __getitem__(self, idx):
        file_name = self.img_list[idx]

        img_path = os.path.join(self.img_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)
        
        # Read fire event raw files, fill NaNs, normalization
        with rio.open(img_path) as img:
            img_data = img.read().astype(np.float32)
            img_data[img_data == img.nodata] = np.nan
            img_data = self.fill_nan(img_data)
            img_data = self.normalize(img_data)
        
        # Read fire masks
        with rio.open(mask_path) as mask:
            mask_data = mask.read()
          
        img_tensor = self.mapping(img_data)
        mask_tensor = torch.from_numpy(mask_data).long().squeeze()

        # Apply transformation
        if self.transforms:
            img_tensor = self.transforms(img_tensor)
            mask_tensor = self.transforms(mask_tensor)

        return img_tensor, mask_tensor

    def mapping(self, data):
        # Prepare image tensor with shape (9, H, W) for Satlas multi-band model
        # Satlas (multi-band) expects [TCI_R, TCI_G, TCI_B, B05, B06, B07, B08, B11, B12]
        # Band order that I saved: [TCI_R, TCI_G, TCI_B, B8, B12]
        H, W = data.shape[1], data.shape[2]
        TCI_R, TCI_G, TCI_B, B8, B12 = data

        if self.multispectral:    
            out = np.zeros((9, H, W), dtype=np.float32)
            out[0] = TCI_R  # Red
            out[1] = TCI_G  # Green
            out[2] = TCI_B  # Blue
            out[6] = B8  # NIR
            out[8] = B12 # SWIR
        else: 
            out = np.zeros((3, H, W), dtype=np.float32)
            out[0] = TCI_R  # Red
            out[1] = TCI_G  # Green
            out[2] = TCI_B  # Blue

        return torch.from_numpy(out).float()
    
    def normalize(self, data):
        # Multispectral: scale each band based on normalization doc
        norm_data = np.zeros_like(data, dtype=np.float32)
        for idx in range(data.shape[0]):
            band = data[idx, :, :]
            if idx < 3:
                norm_band = (band / 255.0).clip(0.0, 1.0)   # RGB
            else:
                norm_band = (band / 8160.0).clip(0.0, 1.0)  # others
            norm_data[idx] = norm_band
        return norm_data


