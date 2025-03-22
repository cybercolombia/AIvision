r"""
In this example, we will show Int8 quantization inference compile and test using LogReg
"""

import argparse
import sys
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from termcolor import colored
from torch.quantization.qconfig import (default_dynamic_qconfig,
                                        default_per_channel_qconfig)

import sambaflow.samba.utils as utils
from sambaflow import samba
from sambaflow.frameworks.quantized.quantized_linear import QuantizedLinear
from sambaflow.samba.lazy_param import lazy_param
from sambaflow.samba.materialize import materialize
from sambaflow.samba.utils.argparser import parse_app_args, parse_yaml_to_args
from sambaflow.samba.utils.benchmark_acc import AccuracyReport
from sambaflow.samba.utils.dataset.mnist import dataset_transform
from sambaflow.samba.utils.pef_utils import get_pefmeta


class LogReg(nn.Module):
    """
    Define the model architecture i.e. the layers in the model and the
    number of features in each layer

    :ivar lin_layer: Linear layer
    :ivar criterion: Cross Entropy loss layer
    """
    def __init__(self, num_features: int, num_classes: int, bias: bool, num_linears: int):
        """

        :param num_features: Number of input features for the model
        :param num_classes: Number of output labels the model classifies inputs
        """
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_linear = num_linears

        #: Hidden layers
        layers = []
        for i in range(self.num_linear - 1):
            layers.append(nn.Linear(in_features=num_features, out_features=num_features // 2, bias=bias))
            num_features = num_features // 2
        self.hidden_layers = nn.ModuleList(layers)

        #: Linear layer for predicting target class of inputs
        self.lin_layer = nn.Linear(in_features=num_features, out_features=num_classes, bias=bias)

        #: Cross Entropy layer for loss computation
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model for the given inputs. The forward pass
        predicts the class labels for the inputs and computes the loss
        between the correct and predicted class labels.

        :param inputs: The input samples in the dataset
        :type inputs: torch.Tensor

        :param targets: The correct labels for the inputs
        :type targets: torch.Tensor

        :return: The loss and predicted classes of the inputs
        :rtype: torch.Tensor
        """
        out = inputs
        for i in range(self.num_linear - 1):
            out = self.hidden_layers[i](out)
        out = self.lin_layer(out)

        loss = self.criterion(out, targets)
        return loss, out


def add_args(parser: argparse.ArgumentParser) -> None:
    # Add DaaS args
    parser.add_argument('--pod-name', type=str, default='starters', help="Pod name the app belongs to")
    parser.add_argument('--script', type=str, default=__file__, help="Script file to run the app")
    parser.add_argument('--mpirun', action='store_true', help="Whether it run with MPIRUN or not")
    parser.add_argument('--world-size', type=int, default=1, help="Number of communicators to run DP/MP apps")
    parser.add_argument('--task-name', type=str, default='classification', help="Task name")
    parser.add_argument('--model-name', type=str, default=__file__.split('/')[-1].split('.')[0], help="Dataset name")
    parser.add_argument('--data-name', type=str, default='mnist', help="Dataset name")
    parser.add_argument('--data-type', type=str, default='image', help="Training dataset type")
    parser.add_argument('--label-type',
                        type=str,
                        default='classes',
                        choices=['classes', 'numbers', 'resolvers'],
                        help="Label type")
    # checkpoint handling arguments
    parser.add_argument('--ckpt-load', action='store_true', help="Load trained and/or quantized checkpoint")
    parser.add_argument('--ckpt-save', action='store_true', help="Save trained and/or quantized checkpoint in ckpt-dir")
    parser.add_argument('--ckpt-dir-bf16', type=str, default='', help="Checkpoint directory for bf16")
    parser.add_argument('--ckpt-dir-int8', type=str, default='', help="Checkpoint directory for int8")
    parser.add_argument('--ckpt-quantize',
                        action='store_true',
                        default='',
                        help="Quantize checkpoint to lower precision")
    parser.add_argument('--quantize-dtype', type=str, default='int8', help="Quantization data type")
    parser.add_argument('--quantize-scheme',
                        type=str,
                        default='per_tensor',
                        help="Quantization scheme: per_tensor vs per_channel")
    parser.add_argument('--dataloader',
                        type=str,
                        default='torchloader',
                        choices=['torchloader', 'sambaloader'],
                        help="Dataloader type")
    parser.add_argument('--logger-name', type=str, default='default', choices=['default', 'acp'], help="Logger type")
    parser.add_argument('--logger-dir', type=str, default='output', help="Logger output dir")

    # Other args
    parser.add_argument('--optim', type=str, default='sgd', help="Optimizer to use for training")
    parser.add_argument('--lr', type=float, default=0.0015, help="Learning rate for training")
    parser.add_argument('--momentum', type=float, default=0.0, help="Momentum value for training")
    parser.add_argument('--weight-decay', type=float, default=3e-4, help="Weight decay for training")
    parser.add_argument('--num-epochs', '-e', type=int, default=1)
    parser.add_argument('--num-steps', type=int, default=-1)
    parser.add_argument('--num-features', type=int, default=784)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--weight-norm', action="store_true", help="Enable weight normalization")
    parser.add_argument('--acc-test', action='store_true', help='Option for accuracy guard test in RDU regression.')
    parser.add_argument('--yaml-config', default=None, type=str, help='YAML file used with launch_app.py')
    parser.add_argument('--data-dir',
                        '--data-folder',
                        type=str,
                        default='mnist_data',
                        help="The folder to download the MNIST dataset to.")
    parser.add_argument('--visualize', action="store_true", help="Generate plots for accuracy comparisons")
    parser.add_argument('--enable-profiler', action='store_true', help='Enable Samba Profiler')
    parser.add_argument('--profiler-trace', type=str, help='Samba profiler trace output file')
    parser.add_argument('--bias', action='store_true', help='Linear layer will learn an additive bias')
    parser.add_argument('--num-linears', type=int, default=1, help='num_linear_layers')
    # end args


def prepare_dataloader(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prep work to train the logreg model with the `MNIST dataset <http://yann.lecun.com/exdb/mnist/>`__:

    We'll split the dataset into train and test sets and return the corresponding data loaders

    :param args: argument specifying the location of the dataset
    :type args: argparse.Namespace

    :return: Train and test data loaders
    :rtype: Tuple[torch.utils.data.DataLoader]
    """

    # Get the train & test data (images and labels) from the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=f'{args.data_dir}',
                                               train=True,
                                               transform=dataset_transform(vars(args)),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root=f'{args.data_dir}',
                                              train=False,
                                              transform=dataset_transform(vars(args)))

    # Get the train & test data loaders (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader


def train(args: argparse.Namespace, model: nn.Module, output_tensors: Tuple[samba.SambaTensor]) -> None:
    """
    Train the model.
    At the end of a training loop, the model will be able
    to correctly predict the class labels for any input, within a certain
    accuracy.

    :param args: Hyperparameter values and accuracy test behavior controls
    :type args: argparse.Namespace

    :param model: Model to be trained
    :type model: torch.nn.Module

    """

    # Get data loaders for training and test data
    train_loader, test_loader = prepare_dataloader(args)

    # Total training steps (iterations) per epoch
    total_step = len(train_loader)

    hyperparam_dict = {"lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay}

    # Train and test for specified number of epochs
    for epoch in range(args.num_epochs):
        avg_loss = 0

        # Train the model for all samples in the train data loader
        for i, (images, labels) in enumerate(train_loader):
            global_step = epoch * total_step + i
            if args.num_steps > 0 and global_step >= args.num_steps:
                print('Maximum num of steps reached. ')
                return None

            data_event = samba.session.profiler.start_event('data_step')
            sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
            sn_labels = samba.from_torch_tensor(labels, name='label', batch_dim=0)
            samba.session.profiler.end_event(data_event)

            compute_step = samba.session.profiler.start_event('compute_step')
            loss, outputs = samba.session.run(input_tensors=[sn_images, sn_labels],
                                              output_tensors=output_tensors,
                                              hyperparam_dict=hyperparam_dict,
                                              data_parallel=args.data_parallel,
                                              reduce_on_rdu=args.reduce_on_rdu)
            samba.session.profiler.end_event(compute_step)

            get_outputs_event = samba.session.profiler.start_event('get_outputs')
            # Sync the loss and outputs with host memory
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            avg_loss += loss.mean()
            samba.session.profiler.end_event(get_outputs_event)

            # Print loss per 10,000th sample in every epoch
            if (i + 1) % 10000 == 0 and args.local_rank <= 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, i + 1, total_step,
                                                                         avg_loss / (i + 1)))

        # Check the accuracy of the trained model for all samples in the test data loader
        # Sync the model parameters with host memory
        samba.session.to_cpu(model)
        test_acc = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for images, labels in test_loader:
                loss, outputs = model(images, labels)
                loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
                total_loss += loss.mean()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            test_acc = 100.0 * correct / total

            if args.local_rank <= 0:
                print('Test Accuracy: {:.2f}'.format(test_acc),
                      ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))

        if args.acc_test:
            assert args.num_epochs == 1, "Accuracy test only supported for 1 epoch"
            assert test_acc > 91.0 and test_acc < 92.0, "Test accuracy not within specified bounds."

    if args.acc_report_json is not None:
        val_metrics = {'acc': test_acc.item() / 100.0, 'loss': total_loss.item() / len(test_loader)}
        report = AccuracyReport(val_metrics=val_metrics,
                                batch_size=args.batch_size,
                                num_iterations=args.num_epochs * total_step)
        report.save(args.acc_report_json)

    if args.ckpt_save:
        # save learnable parameters
        torch.save(model.state_dict(), args.ckpt_dir_bf16)


def int8_inference_test(args: argparse.Namespace, model: nn.Module, output_tensors: Tuple[samba.SambaTensor]) -> None:
    """
    Evaluate the int8 model.
    """

    # Get data loaders for training and test data
    _, test_loader = prepare_dataloader(args)

    test_acc = 0.0
    correct = 0
    total = 0
    total_loss = 0
    for images, labels in test_loader:
        sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
        sn_labels = samba.from_torch_tensor(labels, name='label', batch_dim=0)
        loss, outputs = samba.session.run(input_tensors=[sn_images, sn_labels], output_tensors=output_tensors)
        if total == 0:
            dequantized_tensor = samba.to_torch(
                samba.session.get_tensors_by_name(["logreg__lin_layer__dequantize__outputs__0"])[0])
            original_tensor = torch.load(args.ckpt_dir_bf16)['lin_layer.weight']
            utils.assert_close(dequantized_tensor,
                               original_tensor,
                               "Quantization error per element for lin_layer.weight",
                               threshold=1e-3)

        loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
        total_loss += loss.mean()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    test_acc = 100.0 * correct / total

    if args.local_rank <= 0:
        print('Test Accuracy: {:.2f}'.format(test_acc), ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))

    if args.acc_test:
        assert test_acc > 91.9 and test_acc < 92.0, "Test accuracy of quantized model not within specified bounds."


def quantize(args: argparse.Namespace, model: nn.Module) -> None:
    """
    Quantize trained linear weights to set precision type on CPU

    :param args: Arguments to control
    :type args: argparse.Namespace

    :param model: Model instance
    :type model: torch.nn.Module
    """
    if args.quantize_dtype == "int8":
        quantize_dtype = torch.qint8
    if args.quantize_scheme == "per_tensor":
        quantize_scheme = default_dynamic_qconfig
    elif args.quantize_scheme == "per_channel":
        quantize_scheme = default_per_channel_qconfig

    # Quantize to target dtype using suggested scheme
    quantized_model = torch.quantization.quantize_dynamic(model,
                                                          qconfig_spec={torch.nn.Linear: quantize_scheme},
                                                          dtype=quantize_dtype,
                                                          mapping={torch.nn.Linear: QuantizedLinear})

    # Check that the linear's weight is a quantized tensor
    for quant_linear in [quantized_model.lin_layer]:
        assert quant_linear.weight.is_quantized, "Quantized model's weight is not quantized"

    with materialize(quantized_model) as quantized_model:
        if args.ckpt_save:
            # save int8 parameters
            torch.save(quantized_model.state_dict(), args.ckpt_dir_int8)


def main(argv):
    """
    :param argv: Command line arguments (`compile`, `test`, `run`, `measure-performance` or `measure-sections`)
    """
    utils.set_seed(256)

    migrated_arguments = {'-e': '--num-epochs', '--data-folder': '--data-dir'}
    deprecated_args_found = set(migrated_arguments.keys()) & set(argv)
    if deprecated_args_found:
        for arg in deprecated_args_found:
            print(
                colored(f'This specific argument {arg} is being deprecated, new argument is {migrated_arguments[arg]}',
                        'red'))

    args_cli = parse_app_args(argv=argv, common_parser_fn=add_args)
    args_composed = parse_yaml_to_args(args_cli.yaml_config, args_cli) if args_cli.yaml_config else args_cli

    args = args_composed
    # when it is not distributed mode, local rank is -1.
    args.local_rank = dist.get_rank() if dist.is_initialized() else -1

    # Create random input and output data for testing
    ipt = samba.randn(args.batch_size, args.num_features, name='image', batch_dim=0,
                      named_dims=('B', 'F')).bfloat16().float()
    tgt = samba.randint(args.num_classes, (args.batch_size, ), name='label', batch_dim=0, named_dims=('B', ))

    ipt.host_memory = False
    tgt.host_memory = False

    # Instantiate the model
    model = LogReg(args.num_features, args.num_classes, args.bias, args.num_linears)

    # Create a randomly initialized quantized model if running quantized inference
    if args.inference and args.ckpt_quantize:
        if args.quantize_scheme == "per_tensor":
            quantize_scheme = default_dynamic_qconfig
        elif args.quantize_scheme == "per_channel":
            quantize_scheme = default_per_channel_qconfig
        cm = lazy_param(quantized=True)
        with cm:
            quantized_model = torch.quantization.quantize_dynamic(model,
                                                                  qconfig_spec={torch.nn.Linear: quantize_scheme},
                                                                  dtype=torch.qint8,
                                                                  mapping={torch.nn.Linear: QuantizedLinear})

    # Sync model parameters with RDU memory
    if args.inference and args.ckpt_quantize:
        samba.from_torch_model_(quantized_model)
    else:
        samba.from_torch_model_(model)

    # Annotate parameters if weight normalization is on
    if args.weight_norm:
        utils.weight_norm_(model.lin_layer)

    inputs = (ipt, tgt)

    # Instantiate an optimizer if the model will be trained
    if args.inference:
        optimizer = None
    else:
        # We use the SGD optimizer to update the weights of the model
        optimizer = samba.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.command == "compile":
        #  Compile the model to generate a PEF (Plasticine Executable Format) binary
        #  If --ckpt-quantize, use quantized model for compilation
        if args.inference and args.ckpt_quantize:
            samba.session.compile(quantized_model,
                                  inputs,
                                  optimizer,
                                  name='logreg_torch_quantized',
                                  config_dict=vars(args),
                                  pef_metadata=get_pefmeta(args, quantized_model))
        else:
            samba.session.compile(model,
                                  inputs,
                                  optimizer,
                                  name='logreg_torch',
                                  config_dict=vars(args),
                                  pef_metadata=get_pefmeta(args, model))

    elif args.command in ["test", "run"]:
        if args.enable_profiler:
            samba.session.start_samba_profile()

        # Trace the compiled graph to initialize the model weights and input/output tensors
        # for execution on the RDU.
        if args.command == "test" and args.ckpt_quantize:
            # load quantized ckpt
            quantized_model.load_state_dict(torch.load(args.ckpt_dir_int8))
            traced_outputs = utils.trace_graph(quantized_model,
                                               inputs,
                                               optimizer,
                                               pef=args.pef,
                                               mapping=args.mapping,
                                               distlearn_config=args.distlearn_config,
                                               dev_mode=True)
        else:
            traced_outputs = utils.trace_graph(model,
                                               inputs,
                                               optimizer,
                                               pef=args.pef,
                                               mapping=args.mapping,
                                               distlearn_config=args.distlearn_config)

        if args.command == "test":
            # Test the model's functional correctness. This tests if the result of execution
            # on the RDU is comparable to that on a CPU. CPU run results are used as reference.
            if args.ckpt_quantize:
                test_model = quantized_model
                int8_inference_test(args, test_model, traced_outputs)

        elif args.command == "run":
            # Train the model on RDU. This is where the model will be trained
            # i.e. weights will be learned to fit the input dataset
            train(args, model, traced_outputs)
            # Post Training Quantization (PTQ) to create quantized checkpoint
            if args.ckpt_quantize and args.ckpt_save:
                # Clean up RDU trained model to make it torch cpu runnable
                # instead of samba cpu runnable
                torch_only_state_dict = {
                    param_tensor: model.state_dict()[param_tensor].torch_tensor()
                    for param_tensor in model.state_dict()
                }
                torch_only_model = LogReg(args.num_features, args.num_classes, args.bias, args.num_linears)
                torch_only_model.load_state_dict(torch_only_state_dict)
                quantize(args, torch_only_model)

        if args.enable_profiler:
            samba.session.end_samba_profile(filename=args.profiler_trace)


if __name__ == '__main__':
    main(sys.argv[1:])
