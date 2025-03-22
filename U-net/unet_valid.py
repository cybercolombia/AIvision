import os
import natsort
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import JaccardIndex

from model import Unet
from load import CustomDataset


import sys

#Loss functions from: https://github.com/JunMa11/SegLoss
sys.path.append('SegLoss-master/losses_pytorch')
from dice_loss import IoULoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

#parameters
H = 128#256
W = 128#256
size = (H,W)
batch_size = 2

#LOAD DATA------------------------
#Define a transform object that takes the data to pytorch tensor form and normalizes it
img_transform = Compose( [Resize(size), ToTensor()] )#, Normalize(mean=(0.5,),std=(0.5,))] );
msk_transform = Compose( [Resize(size)] );

train_set = CustomDataset(main_dir='./data/train/',img_transform=img_transform,msk_transform=msk_transform)
valid_set = CustomDataset(main_dir='./data/test/',img_transform=img_transform,msk_transform=msk_transform)

print("Dataset size:\nValidation: {0}".format(len(valid_set)))

img, msk = valid_set[0]
print(img.shape, msk.shape)

valid_loader = DataLoader(
    dataset = valid_set,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 2
)

#To load the saved version of the model
saved_model = Unet()
saved_model.load_state_dict(torch.load('./model_20250314_084052_20'))

it_val = iter(valid_loader)
iou_av = 0
iou_sq = 0
cnt = 0
for it in it_val:
    input_im, masks = it
    #print(type(input_im[0]), input_im[0].shape)
    #plt.imshow(input_im[0].permute(1,2,0))
    #plt.imshow(masks[0][0])
    
    out = saved_model(input_im)
    #print(type(out[0][0]), out[0][0].shape)
    #plt.imshow(out[0][0].detach().numpy())
    ##Jackard index (IoU score)
    jaccard = JaccardIndex(task='multiclass',num_classes=2)
    iou_sc = jaccard(out[0][0], masks[0][0])
    iou_av += iou_sc.item()
    iou_sq += iou_sc.item()**2
    cnt += 1
iou_av /= cnt
iou_fl = np.sqrt(iou_sq/cnt - iou_av**2)
print("IoU score: {0:.3f}+/-{1:.3f}".format(iou_av,iou_fl))
