import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.optim as optim
import sambaflow.samba.utils as utils
#from sambaflow.samba.utils.common import common_app_driver
from sambaflow.samba.utils.pef_utils import get_pefmeta
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.sambaloader import SambaLoader

import sys
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# The model stays the same
class ConvNet(nn.Module):
    """
    Instantiate a 4-layer CNN for MNIST Image Classification.

    In SambaFlow, it is possible to include a loss function as part of a model's definition and put it in
    the forward method to be computed.

    Typical SambaFlow usage example:

    model = ConvNet()
    samba.from_torch_model_(model)
    optimizer = ...
    inputs = ...
    if args.command == "run":
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        train(args, model)
    """

    def __init__(self):

        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.criterion = nn.CrossEntropyLoss() # Add loss function to model

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        loss = self.criterion(out, labels)     # Compute loss
        return loss, out

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
        samba.randn(args.bs, 1, 28, 28, name="image", batch_dim=0),
        samba.randint(args.num_classes, (args.bs,), name="label", batch_dim=0),
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
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Get the train & test data (images and labels) from the MNIST dataset
    train_dataset = datasets.MNIST(
        root=args.data_path,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=transform)

    # Set up the train & test data loaders (input pipeline)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.bs, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.bs, shuffle=False
    )

    # Create SambaLoaders
    sn_train_loader = SambaLoader(train_loader, ["image", "label"])
    sn_test_loader = SambaLoader(test_loader, ["image", "label"])

    return sn_train_loader, sn_test_loader

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
        for i, (images, labels) in enumerate(sn_train_loader):

            # Run the model on RDU: forward -> loss/gradients -> backward/optimizer
            loss, outputs = samba.session.run(
                input_tensors=(images, labels),
                output_tensors=model.output_tensors,
                hyperparam_dict=hyperparam_dict
            )

            # Convert SambaTensors back to Torch Tensors to calculate accuracy
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            loss_list.append(loss.tolist())

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(
                        epoch + 1,
                        args.num_epochs,
                        i + 1,
                        total_step,
                        torch.mean(loss),
                        (correct / total) * 100,
                    )
                )

def main(argv):

    args = parse_app_args(argv=argv, common_parser_fn=add_user_args)

    # Create the CNN model
    model = ConvNet()

    # Convert model to SambaFlow (SambaTensors)
    samba.from_torch_model_(model)

    # Create optimizer
    # Note that SambaFlow currently supports AdamW, not Adam, as an optimizer
    optimizer = samba.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Normally, we'd define a loss function here, but with SambaFlow, it can be defined
    # as part of the model, which we have done in this case

    # Create dummy SambaTensor for graph tracing
    inputs = get_inputs(args)

    # The common_app_driver() handles model compilation and various other tasks, e.g.,
    # measure-performance.  Running, or training, a model must be explicitly carried out
    if args.command == "run":
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        train(args, model)
    else:
        # common_app_driver(args=args,
        #                 model=model,
        #                 inputs=inputs,
        #                 optim=optimizer,
        #                 name=model.__class__.__name__,
        #                 init_output_grads=not args.inference,
        #                 app_dir=utils.get_file_dir(__file__))
        samba.session.compile(model=model,
                              inputs=inputs,
                              optimizers=optimizer,
                              name='convnet_torch',
                              config_dict=vars(args),
                              pef_metadata=get_pefmeta(args, model))

if __name__ == '__main__':
    main(sys.argv[1:])