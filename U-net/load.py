import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

class CustomDataset(Dataset):
    def __init__(self, main_dir, img_transform, msk_transform):
        self.main_dir = main_dir
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.img_dir = main_dir+'images/'
        self.msk_dir = main_dir+'masks/'
        all_images = os.listdir(self.img_dir) 
        all_masks = os.listdir(self.msk_dir)
        all_images.sort()
        all_masks.sort()
        self.images = all_images
        self.masks = all_masks
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.images[idx])
        img = Image.open(img_loc).convert('RGB')
        tensor_img = self.img_transform(img)
        msk_loc = os.path.join(self.msk_dir, self.masks[idx])
        msk = Image.open(msk_loc).convert('L') 
        tensor_msk = pil_to_tensor(self.msk_transform(msk))
        tensor_msk = torch.where(tensor_msk == 255, 1, 0) #values 0 or 1 
        return tensor_img, tensor_msk


class ImageData(Dataset):
    def __init__(self, directory, transform):
         self.directory = directory
         self.transform = transform
         images = os.listdir(self.directory)
         images.sort()
         self.images = images
         
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.directory, self.images[idx])
        img = Image.open(img_loc).convert('RGB')
        tensor_img = self.transform(img)
        return tensor_img

    
         
