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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np

from model import AutoEncoder


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
        "-bst",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 100)",
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
        help="Download location for MNIST data",
    )
    parser.add_argument(
        "--model-path", type=str, default="model", help="Save location for model"
    )

def get_inputs(args: argparse.Namespace) -> Tuple[samba.SambaTensor]:
    """
    Generates random SambaTensors in the same shape as image  and mask tensors.

    In order to properly compile a PEF and trace the model graph, SambaFlow requires a SambaTensor that
    is the same shape as the input Torch Tensors, allowing the graph to be optimally mapped onto an RDU.

    Args:
        args (argparse.Namespace): User- and system-defined command line arguments

    Returns:
        A tuple of SambaTensors with random values in the same shape as MNIST image and label tensors.
    """

    dummy_image = (
        samba.randn(args.bs, 28*28, name="image", batch_dim=0),
    )

    return dummy_image

def prepare_dataloader(args: argparse.Namespace) -> Tuple[sambaflow.samba.sambaloader.SambaLoader, sambaflow.samba.sambaloader.SambaLoader]:
    """
    Transforms MNIST input to tensors and creates training/test dataloaders.

    Downloads the MNIST dataset (if necessary); splits the data into training and test sets; transforms the
    data to tensors; then creates Torch DataLoaders over those sets.  Torch DataLoaders are wrapped in
    SambaLoaders.

    Args:
        args (argparse.Namespace): User- and system-defined command line arguments

    Returns:
        A tuple of SambaLoaders over the training and test sets.
    """

    # Transform the raw MNIST data into PyTorch Tensors, which will be converted to SambaTensors
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Get the train & test data (images and labels) from the MNIST dataset
    train_dataset = datasets.MNIST(root=args.data_path, train=True, transform=trans, download=True)
    test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=trans)

    # Set up the train & test data loaders (input pipeline)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bst, shuffle=False)

    # Create SambaLoaders
    sn_train_loader = SambaLoader(train_loader, ["image", "label"])
    sn_test_loader = SambaLoader(test_loader, ["image", "label"])

    return sn_train_loader, sn_test_loader

def train(args: argparse.Namespace,
          epoch: int, log_interval: int,
          model: nn.Module, sn_train_loader: SambaLoader,
          train_losses: List[float], train_counter: List[int]) -> None:
    hyperparam_dict = {"lr": args.learning_rate}
    
    model.train() #tell the model it is a training round
    for batch_idx, data in enumerate(sn_train_loader):
        x_batch, _ = data
        x_batch = x_batch.reshape(-1, 28*28)
        
        loss, pred = samba.session.run(
            imput_tensors = (x_batch,),
            output_tensors=model.output_tensors,
            hyperparam_dict=hyperparam_dict
        )
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(x_batch), len(train_loader.dataset),
                100.*batch_idx/len(train_loader), loss.item()
            ))
            train_losses.append(loss.item())
            train_counter.append(batch_idx*args.bs + epoch*len(train_loader.dataset))

def test(model: nn.Module, sn_test_loader: SambaLoader,
        test_losses: List[float]) -> None:
    model.eval() #tell the model it is being evaluated
    test_loss = 0
    with torch.no_grad():
        for data in sn_test_loader:
            x_batch, _ = data
            x_batch = x_batch.reshape(-1, 28*28)
            loss, pred = model(x_batch)
            test_loss += loss.item()
            val = pred.data.max(1, keepdim=True)[1]
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}'.format(test_loss,))

def main(argv):
    args = parse_app_args(argv=argv, common_parser_fn=add_user_args)
    log_interval = 100
    
    # Create the model
    model = AutoEncoder()

    # Convert model to SambaFlow (SambaTensors)
    samba.from_torch_model_(model)

    # Create optimizer
    optimizer = samba.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Create dummy SambaTensor for graph tracing
    inputs = get_inputs(args)

    # measure-performance.  Running, or training, a model must be explicitly carried out
    if args.command == "run":
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        sn_train_loader, sn_test_loader = prepare_dataloader(args)

        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]
        
        test(model, sn_test_loader, test_losses)
        for epoch in range(args.num_epochs):
            train(args, epoch, log_interval,
                  model,
                  sn_train_loader,
                  train_losses, train_counter)
            test(model, sn_test_loader, test_losses)
    else:
        samba.session.compile(model=model,
                              inputs=inputs,
                              optimizers=optimizer,
                              name='aenet_torch',
                              config_dict=vars(args),
                              pef_metadata=get_pefmeta(args, model))

if __name__ == '__main__':
    main(sys.argv[1:])