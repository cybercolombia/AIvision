import os
#import natsort
import random
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
#from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
#from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
#from torchvision.transforms.functional import pil_to_tensor
#from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, List

from model import Unet

WORKING_DIR = '/home/carlos/Documents/CyberComputingConsulting/AIvision/repo/AIvision/U-net/'
MODEL_STORE_PATH = WORKING_DIR + 'model_store/'
DATA_FOLDER = 'data2/' #'data/'
IMG_HEIGHT = 128 #512
IMG_WIDTH = 128 #512
 

#Set seed for reproducibility
def seeding(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#LOAD DATA----------------------------------
class CaravanaDataset(Dataset):
    def __init__(self, root_path: str, limit: int=None):
        self.root_path = root_path
        self.transform = Compose([Resize((IMG_HEIGHT, IMG_WIDTH)), ToTensor()])

        self.images = sorted([root_path + "train_images/" + i for i in os.listdir(root_path + "train_images/")])
        self.masks = sorted([root_path + "train_masks/" + i for i in os.listdir(root_path + "train_masks/")])

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
        

def prepare_dataloader(batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = CaravanaDataset(WORKING_DIR+DATA_FOLDER, limit=200) #REMOVE LIMIT FOR FULL TRAINING!!!!!!
    print("Dataset size: ",len(dataset))

    generator = torch.Generator().manual_seed(25)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)
    
    train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    return train_loader, val_loader, test_loader

#DICE coefficient = 2*(A n B) / (|A|+|B|)
def dice_coefficient(prediction: torch.Tensor, target: torch.Tensor, 
                     epsilon: float=1e-07) -> float:
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice

#TRAINING-----------------------------------
def train(num_epochs: int, model: Unet, train_loader: DataLoader, 
          val_loader: DataLoader, optimizer: torch.optim.AdamW, 
          device: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []

    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0
        train_running_dc = 0

        for idx, data in enumerate(train_loader):
            img = data[0].to(device)
            mask = data[1].to(device)

            loss, y_pred = model(img, mask)
            dc = dice_coefficient(y_pred, mask)
            optimizer.zero_grad()
            
            train_running_loss += loss.item()
            train_running_dc += dc.item()

            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print('Epoch [{}/{}], Step {}, Loss: {:.4f}'.format(epoch+1, num_epochs, idx, loss.item()))
            
            torch.cuda.empty_cache()

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)

        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        model.eval()
        val_running_loss = 0
        val_running_dc = 0

        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                img = data[0].to(device)
                mask = data[1].to(device)
                
                loss, y_pred = model(img, mask)
                dc = dice_coefficient(y_pred, mask)
            
                val_running_loss += loss.item()
                val_running_dc += dc.item()
            
            val_loss = val_running_loss / (idx + 1)
            val_dc = val_running_dc / (idx + 1)

        val_losses.append(val_loss)
        val_dcs.append(val_dc)

        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
        print("\n")
        print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
        print("-" * 30)
    
    return train_losses, train_dcs, val_losses, val_dcs



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    #parameters
    batch_size = 2 #8
    num_epochs = 50
    learn_rate = 3.e-4

    seeding(123)

    #LOAD DATA------------------------
    train_loader, val_loader, test_loader = prepare_dataloader(batch_size)
    
    #THE MODEL-----------------
    model = Unet().to(device)

    #OPTIMIZER-------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)
    
    #TRAIN--------------------------------
    train_losses, train_dcs, val_losses, val_dcs = train(num_epochs, model, train_loader, val_loader, optimizer, device)

    #SAVE MODEL
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODEL_STORE_PATH + 'model_{}'.format(timestamp)
    torch.save(model, model_path)

if __name__ == '__main__':
    main()