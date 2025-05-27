import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.optim as optim
import sambaflow.samba.utils as utils
from sambaflow.samba.utils.pef_utils import get_pefmeta
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.sambaloader import SambaLoader

import sys
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import TestModel


# The model stays the same

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
        help="Location of data",
    )
    parser.add_argument(
        "--model-path", type=str, default="model", help="Save location for model"
    )

def get_inputs(args: argparse.Namespace) -> Tuple[samba.SambaTensor]:
    """
    Generates random SambaTensors in the same shape as data  and label tensors.

    In order to properly compile a PEF and trace the model graph, SambaFlow requires a SambaTensor that
    is the same shape as the input Torch Tensors, allowing the graph to be optimally mapped onto an RDU.

    Args:
        args (argparse.Namespace): User- and system-defined command line arguments

    Returns:
        A tuple of SambaTensors with random values in the same shape as data and label tensors.
    """

    dummy_data = (
        samba.randn(args.bs, 5, name="data", batch_dim=0),
    )

    return dummy_data

class CustomDataset(Dataset):
    def __init__(self, data_file, targets_file=None):
        self.data_file = data_file
        self.targets_file = targets_file
        self.data, self.targets = self._read_data()

    def _read_data(self):
        data = np.load(self.data_file)
        targets = np.load(self.targets_file) if self.targets_file else None
        if targets is None:
            # If no targets file is provided, create dummy targets
            targets = np.zeros(data.shape[0], dtype=np.int64)
        # Ensure data is in the correct format
        data = data.astype(np.float32)
        targets = targets.astype(np.int64)
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_data = torch.tensor(self.data[idx], dtype=torch.float32)
        sample_target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sample_data, sample_target

def prepare_dataloader(args: argparse.Namespace) -> Tuple[SambaLoader]:
    # Data set
    train_dataset = CustomDataset(args.data_path + "X.npy", DATA_PATH + "y.npy")
    print("train_dataset size:", len(train_dataset))
    print("train_dataset data shape:", train_dataset.data.shape)
    print("train_dataset targets shape:", train_dataset.targets.shape)
    # Check if the dataset is empty
    if len(train_dataset) == 0:
        raise ValueError("The dataset is empty. Please check the data files.")
    
    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Create SambaLoaders
    sn_train_loader = SambaLoader(train_loader, ["data", "labels"])
        
    return sn_train_loader

def train(args: argparse.Namespace, model: nn.Module) -> None:
    """
    Trains the model.

    Prepares and loads the data, then runs the training loop with the hyperparameters specified
    by the input arguments.  Calculates loss and accuracy over the course of training.

    Args:
        args (argparse.Namespace): User- and system-defined command line arguments
        model (nn.Module): ConvNet model
    """

    sn_train_loader, _ = prepare_dataloader(args)
    hyperparam_dict = {"lr": args.learning_rate}

    total_step = len(sn_train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(args.num_epochs):
        for i, (points, labels) in enumerate(sn_train_loader):

            # Run the model on RDU: forward -> loss/gradients -> backward/optimizer
            loss, outputs = samba.session.run(
                input_tensors=(points, labels),
                output_tensors=model.output_tensors,
                hyperparam_dict=hyperparam_dict
            )

            # Convert SambaTensors back to Torch Tensors to calculate accuracy
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            loss_list.append(loss.tolist())

            if (i + 1) % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1,
                        args.num_epochs,
                        i + 1,
                        total_step,
                        torch.mean(loss),
                    )
                )

def main(argv):
    # Hyperparameters
    args = parse_app_args(argv=argv, common_parser_fn=add_user_args)

    # Create the model
    model = TestModel()

    # Convert model to SambaFlow (SambaTensors)
    samba.from_torch_model_(model)

    # Create optimizer
    optimizer = samba.optim.SGD(model.parameters(), lr=args.learning_rate)

    # Normally, we'd define a loss function here, but with SambaFlow, it can be defined
    # as part of the model, which we have done in this case
    
    # Create dummy SambaTensor for graph tracing
    inputs = get_inputs(args)

    if args.command == "run":  # run the training
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        train(args, model)
    else:                      # compile 
        samba.session.compile(model=model,
                              inputs=inputs,
                              optimizers=optimizer,
                              name='logclass',
                              config_dict=vars(args),
                              pef_metadata=get_pefmeta(args, model))
    
if __name__ == '__main__':
    main(sys.argv[1:])