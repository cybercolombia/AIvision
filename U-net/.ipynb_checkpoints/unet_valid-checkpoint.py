import os
#import natsort
import random
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
#from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
#from torchvision.transforms.functional import pil_to_tensor
#from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import JaccardIndex

#from model import Unet
#from load import CustomDataset

import sys

WORKING_DIR = '/home/carlos/Documents/CyberComputingConsulting/AIvision/repo/AIvision/U-net/'
MODEL_STORE_PATH = WORKING_DIR + 'model_store/'
DATA_FOLDER = 'data2/' #'data/'
IMG_HEIGHT = 128 #512
IMG_WIDTH = 128 #512

##Loss functions from: https://github.com/JunMa11/SegLoss
#sys.path.append('SegLoss-master/losses_pytorch')
#from dice_loss import IoULoss

#LOAD DATA----------------------------------
class CaravanaDataset(Dataset):
    def __init__(self, root_path: str, limit: int=None):
        self.root_path = root_path
        self.transform = Compose([Resize((IMG_HEIGHT, IMG_WIDTH)), ToTensor()])

        self.images = sorted([root_path + "test_images/" + i for i in os.listdir(root_path + "test_images/")])
        self.masks = sorted([root_path + "test_masks/" + i for i in os.listdir(root_path + "test_masks/")])

        if (limit == None) or (limit >= len(self.images)):
            self.limit = len(self.images)
        else:
            self.limit = limit
        self.images = self.images[:self.limit]
        self.masks = self.masks[:self.limit] 

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index]).convert('L')

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")
    
    #parameters
    size = (IMG_HEIGHT,IMG_WIDTH)
    batch_size = 2
    
    #LOAD DATA------------------------
    valid_set = CaravanaDataset(WORKING_DIR+DATA_FOLDER, limit=200)

    print("Validation Dataset size: {0}".format(len(valid_set)))
    
    img, msk = valid_set[0]
    print(img.shape, msk.shape)
    
    valid_loader = DataLoader(
        dataset = valid_set,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 2
    )

    #To load the saved version of the model
    saved_model = torch.load(MODEL_STORE_PATH+'model_20250414_110036',
                             weights_only=False).to(device)
    saved_model.eval()
    
    it_val = iter(valid_loader)
    iou_av = 0
    iou_sq = 0
    cnt = 0
    for it in it_val:
        input_im, masks = it
        input_im = input_im.to(device)
        masks = masks.to(device)
        #print(type(input_im[0]), input_im[0].shape)
        #plt.imshow(input_im[0].permute(1,2,0))
        #plt.imshow(masks[0][0])
        
        _, out = saved_model(input_im, masks)
        out = out.to(device)
        #print(type(out[0][0]), out[0][0].shape)
        #plt.imshow(out[0][0].detach().numpy())
        ##Jackard index (IoU score)
        jaccard = JaccardIndex(task='multiclass',num_classes=2).to(device)
        iou_sc = jaccard(out[0][0], masks[0][0])
        iou_av += iou_sc.item()
        iou_sq += iou_sc.item()**2
        cnt += 1
    iou_av /= cnt
    iou_fl = np.sqrt(iou_sq/cnt - iou_av**2)
    print("IoU score: {0:.3f}+/-{1:.3f}".format(iou_av,iou_fl))
