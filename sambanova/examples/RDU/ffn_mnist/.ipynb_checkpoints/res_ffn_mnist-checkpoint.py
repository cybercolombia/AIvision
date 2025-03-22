#!/usr/bin/python
# encoding: utf-8

# Copyright © 2018-2020 by SambaNova Systems, Inc. Disclosure, reproduction,
# reverse engineering, or any other use made without the advance written
# permission of SambaNova Systems, Inc. is unauthorized and strictly
# prohibited. All rights of ownership and enforcement are reserved.

import argparse
import sys
from typing import List, Tuple

import torch
import torch.nn as nn

import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.utils as utils
import sambaflow.samba.utils.dataset.mnist as mnist
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.common import common_app_driver


def basic_weight_init(module: nn.Module):
    if type(module) is nn.Linear:
        nn.init.xavier_normal_(module.weight)


class ResFFNLogReg(nn.Module):
    """Feed Forward Network with two different activation functions and a residual connection
    """
    def __init__(self, num_features: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.gemm1 = nn.Linear(num_features, hidden_size, bias=True)
        self.gemm2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.gemm3 = nn.Linear(hidden_size, num_classes, bias=True)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.tanh1 = nn.Tanh()
        self.sigmoid1 = nn.Sigmoid()

        self.criterion = nn.CrossEntropyLoss()

        self.apply(basic_weight_init)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor]:
        out = self.gemm1(inputs)
        out = self.norm1(out)
        out = self.tanh1(out)
        residual = out
        out = self.gemm2(out)
        out = self.norm2(out)
        out = out + residual
        out = self.sigmoid1(out)
        out = self.gemm3(out)
        loss = self.criterion(out, targets)
        return loss, out


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('-c', '--num-classes', type=int, default=10)
    parser.add_argument('-e', '--num-epochs', type=int, default=1)
    parser.add_argument('-k', '--num-features', type=int, default=784)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--acc-test', action='store_true', help='Option for accuracy guard test in RDU regression.')


def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument('--log-path', type=str, default='checkpoints')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--dump-interval', type=int, default=1)
    parser.add_argument('--gen-reference', action='store_true', help="Generate PyTorch reference data")
    parser.add_argument('-e', '--num-epochs', type=int, default=1)
    parser.add_argument('-n', '--num-iterations', type=int, default=100, help='Number of iterations to run the pef for')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--measure-train-performance', action='store_true')


def get_inputs(args: argparse.Namespace) -> Tuple[samba.SambaTensor, samba.SambaTensor]:
    image = samba.randn(args.batch_size, args.num_features, name='image', batch_dim=0)
    label = samba.randint(args.num_classes, (args.batch_size, ), name='label', batch_dim=0)
    return image, label


def run_test(args: argparse.Namespace, model: nn.Module, inputs: Tuple[samba.SambaTensor],
             outputs: Tuple[samba.SambaTensor]) -> None:
    outputs_gold = model(*inputs)

    outputs_samba = samba.session.run(input_tensors=inputs, output_tensors=outputs)

    # check that all samba and torch outputs match numerically
    for i, (output_samba, output_gold) in enumerate(zip(outputs_samba, outputs_gold)):
        utils.assert_close(output_samba, output_gold, f'forward output #{i}', threshold=2e-2, visualize=args.visualize)

    if not args.inference:
        torch_loss, torch_gemm_out = outputs_gold
        torch_loss.mean().backward()

        # we test the gradient correctness for all 3 GEMM weights
        gemm1_grad_gold = model.gemm1.weight.grad
        gemm2_grad_gold = model.gemm2.weight.grad
        gemm3_grad_gold = model.gemm3.weight.grad
        gemm1_grad_samba = model.gemm1.weight.sn_grad
        gemm2_grad_samba = model.gemm2.weight.sn_grad
        gemm3_grad_samba = model.gemm3.weight.sn_grad

        utils.assert_close(gemm1_grad_gold,
                           gemm1_grad_samba,
                           'gemm1__weight__grad',
                           threshold=3e-3,
                           visualize=args.visualize)
        utils.assert_close(gemm2_grad_gold,
                           gemm2_grad_samba,
                           'gemm2__weight__grad',
                           threshold=3e-3,
                           visualize=args.visualize)
        utils.assert_close(gemm3_grad_gold,
                           gemm3_grad_samba,
                           'gemm3__weight__grad',
                           threshold=3e-3,
                           visualize=args.visualize)


def train(args: argparse.Namespace, model: nn.Module, optimizer: samba.optim.SGD) -> None:
    train_loader, test_loader = mnist.prepare_dataloader(vars(args))
    # Train the model
    total_step = len(train_loader)
    hyperparam_dict = {"lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay}
    for epoch in range(args.num_epochs):
        avg_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
            sn_labels = samba.from_torch_tensor(labels, name='label', batch_dim=0)

            loss, outputs = samba.session.run(input_tensors=[sn_images, sn_labels],
                                              output_tensors=model.output_tensors,
                                              hyperparam_dict=hyperparam_dict)
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            avg_loss += loss.mean()

            if (i + 1) % 10000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, i + 1, total_step,
                                                                         avg_loss / (i + 1)))

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
            print('Test Accuracy: {:.2f}'.format(test_acc),
                  ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))

        if args.acc_test:
            assert args.num_epochs == 1, "Accuracy test only supported for 1 epoch"
            assert test_acc > 91.0 and test_acc < 92.0, "Test accuracy not within specified bounds."


def main(argv: List[str]):
    utils.set_seed(256)
    args = parse_app_args(argv=argv, common_parser_fn=add_args, run_parser_fn=add_run_args)

    model = ResFFNLogReg(args.num_features, args.hidden_size, args.num_classes)
    samba.from_torch_model_(model)

    optim = None
    if not args.inference:
        optim = sambaflow.samba.optim.SGD(model.parameters(),
                                          lr=args.lr,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay)
    inputs = get_inputs(args)

    common_app_driver(
        args,
        model,
        inputs,
        optim,
        name='resffnlogreg',
    )

    if args.command == "test":
        utils.trace_graph(model, inputs, optim, pef=args.pef, mapping=args.mapping)
        outputs = model.output_tensors
        run_test(args, model, inputs, outputs)
    elif args.command == "run":
        utils.trace_graph(model, inputs, optim, pef=args.pef, mapping=args.mapping)
        train(args, model, optim)


if __name__ == '__main__':
    main(sys.argv[1:])
