import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.optim as optim
import sambaflow.samba.utils as utils
from sambaflow.samba.utils.pef_utils import get_pefmeta
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.sambaloader import SambaLoader

import sys
import argparse
from typing import Tuple, List

import os
import random
import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

from model import Unet


#WORKING_DIR = ''
#MODEL_STORE_PATH = WORKING_DIR + ''


#Include this function in any sambanova conversion
def add_user_args(parser: argparse.ArgumentParser) -> None:
    """
    Add user-defined arguments.

    Args:
        parser (argparse.ArgumentParser): SambaFlow argument parser
    """

    parser.add_argument(
        "-bs",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=6,
        metavar="N",
        help="number of epochs to train (default: 6)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        metavar="N",
        help="number of classes in dataset (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Download location for data",
    )
    parser.add_argument(
        "--model-path", type=str, default="model", help="Save location for model"
    )

#Include this function in any sambanova conversion
def get_inputs(args: argparse.Namespace) -> Tuple[samba.SambaTensor]:
    """
    Generates random SambaTensors in the same shape as MNIST image  and label tensors.

    In order to properly compile a PEF and trace the model graph, SambaFlow requires a SambaTensor that
    is the same shape as the input Torch Tensors, allowing the graph to be optimally mapped onto an RDU.

    Args:
        args (argparse.Namespace): User- and system-defined command line arguments

    Returns:
        A tuple of SambaTensors with random values in the same shape as image and mask tensors.
    """

    dummy_image = (
        samba.randn(args.bs, 3, 512, 512, name="image", batch_dim=0),
        samba.randn(args.bs, 1, 512, 512, name="mask", batch_dim=0),
    )

    return dummy_image

#LOAD DATA----------------------------------
class CaravanaDataset(Dataset):
    def __init__(self, root_path: str, limit: int=None):
        self.root_path = root_path
        self.transform = Compose([Resize((512, 512)), ToTensor()])

        self.images = sorted([root_path + "train_images/" + i for i in os.listdir(root_path + "train_images/")])
        self.masks = sorted([root_path + "train_masks/" + i for i in os.listdir(root_path + "train_masks/")])

        if limit == None:
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
        

def prepare_dataloader(args: argparse.Namespace) -> Tuple[SambaLoader, SambaLoader, SambaLoader]:
    dataset = CaravanaDataset(args.data_path, limit=1000) #REMOVE LIMIT FOR FULL TRAINING!!!!!!
    #print("Dataset size: ",len(dataset))

    # Create datasets
    generator = torch.Generator().manual_seed(25)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.bs,
                                  shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.bs,
                                shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.bs,
                                 shuffle=True)

    # Create SambaLoaders
    sn_train_loader = SambaLoader(train_loader, ["image", "mask"])
    sn_val_loader = SambaLoader(val_loader, ["image", "mask"])
    sn_test_loader = SambaLoader(test_loader, ["image", "mask"])
    
    return sn_train_loader, sn_val_loader, sn_test_loader


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
def train(args: argparse.Namespace, model: nn.Module, sn_train_loader: SambaLoader, 
          sn_val_loader: SambaLoader) -> None:
    train_losses = []
    train_dcs = []
    # val_losses = []
    # val_dcs = []

    hyperparam_dict = {"lr": args.learning_rate}
    
    for epoch in range(args.num_epochs):
        model.train()
        train_running_loss = 0
        train_running_dc = 0

        for idx, data in enumerate(sn_train_loader):
            img, mask = data

            # Run the model on RDU: forward -> loss/gradients -> backward/optimizer
            loss, y_pred = samba.session.run(
                input_tensors=(img, mask),
                output_tensors=model.output_tensors,
                hyperparam_dict=hyperparam_dict
            )

            # Convert SambaTensors back to Torch Tensors to calculate dice coef
            loss, y_pred = samba.to_torch(loss), samba.to_torch(y_pred)
            mask = samba.to_torch(mask)
            dc = dice_coefficient(y_pred, mask)
            
            train_running_loss += loss.item()
            train_running_dc += dc.item()
            
            if idx % 10 == 0:
                print('Epoch [{}/{}], Step {}, Loss: {:.4f}'.format(epoch+1, num_epochs, idx, loss.item()))
            

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)

        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        # model.eval()
        # val_running_loss = 0
        # val_running_dc = 0

        # with torch.no_grad():
        #     for idx, data in enumerate(sn_val_loader):
        #         img, mask = data
                
        #         loss, y_pred = model(img, mask)
        #         dc = dice_coefficient(y_pred, mask)
            
        #         val_running_loss += loss.item()
        #         val_running_dc += dc.item()
            
        #     val_loss = val_running_loss / (idx + 1)
        #     val_dc = val_running_dc / (idx + 1)

        # val_losses.append(val_loss)
        # val_dcs.append(val_dc)

        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
        # print("\n")
        # print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        # print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
        # print("-" * 30)
    
    # return train_losses, train_dcs, val_losses, val_dcs


def main(argv):
    #parameters
    args = parse_app_args(argv=argv, common_parser_fn=add_user_args)
    
    #THE MODEL-----------------
    model = Unet()

    # Convert model to SambaFlow (SambaTensors)
    samba.from_torch_model_(model)

    #OPTIMIZER-------------------------------
    optimizer = samba.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Normally, we'd define a loss function here, but with SambaFlow, it can be defined
    # as part of the model, which we have done in this case

    # Create dummy SambaTensor for graph tracing
    inputs = get_inputs(args)
    
    # The common_app_driver() handles model compilation and various other tasks, e.g.,
    # measure-performance.  Running, or training, a model must be explicitly carried out
    if args.command == "run":
        #LOAD DATA------------------------
        sn_train_loader, sn_val_loader, sn_test_loader = prepare_dataloader(batch_size)
        
        #TRAIN------------------------------------------
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        train(args, model, sn_train_loader, sn_val_loader)

        ##SAVE MODEL
        #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #model_path = MODEL_STORE_PATH + 'model_{}.pth'.format(timestamp)
        #torch.save(model.state_dict(), model_path)
    else:
        #Compile
        samba.session.compile(model=model,
                              inputs=inputs,
                              optimizers=optimizer,
                              name='unet_torch',
                              config_dict=vars(args),
                              pef_metadata=get_pefmeta(args, model))
    
    

if __name__ == '__main__':
    main(sys.argv[1:])