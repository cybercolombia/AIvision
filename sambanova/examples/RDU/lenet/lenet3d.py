import argparse
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from matplotlib.pyplot import cm
from tqdm import tqdm

import sambaflow.samba as samba
import sambaflow.samba.utils as utils
from sambaflow.mac.metadata import ConvTilingMetadata
from sambaflow.samba.env import use_mock_samba_runtime
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.common import common_app_driver

MOCK_SAMBA_RUNTIME = use_mock_samba_runtime()


def add_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument('--in-channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--depth', type=int, default=16, help='Number of depth')
    parser.add_argument('--height', type=int, default=16, help='Number of height')
    parser.add_argument('--width', type=int, default=16, help='Number of width')
    parser.add_argument('--enable-tiling', type=bool, default=False, help='Enable DRAM tiling')

    # Arguments for training.
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--data-file', type=str, default='archive/full_dataset_vectors.h5', help='Dataset for training')
    parser.add_argument('--acc-test', action="store_true", help='Check for accuracy test')


def get_inputs(args: argparse.Namespace) -> Tuple[samba.SambaTensor, ...]:
    images = samba.randn(args.batch_size,
                         args.in_channels,
                         args.depth,
                         args.height,
                         args.width,
                         name='image',
                         batch_dim=0)
    labels = samba.randint(10, (args.batch_size, ), name='label', batch_dim=0)
    if not args.inference:
        images.requires_grad_(True)
    return (images, labels)


class Lenet3D(nn.Module):
    def __init__(self):
        super(Lenet3D, self).__init__()
        # TODO: Add maxpool3d instead of strides when it works.
        self.conv_layer1 = self._conv_layer_set(3, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(4**3 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.LeakyReLU()
        self.ce = nn.CrossEntropyLoss()

    def _conv_layer_set(self, in_c: int, out_c: int) -> nn.Module:
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(2, 2, 2), padding=0, stride=2, bias=False),
            nn.ReLU(),
        )
        return conv_layer

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        loss = self.ce(out, labels)
        return loss


def prepare_dataloader(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader]:
    with h5py.File(args.data_file, 'r') as dataset:
        x_train = dataset["X_train"][:]
        y_train = dataset["y_train"][:]

    num_samples = x_train.shape[0]
    xtrain = np.ndarray((num_samples, 4096, 3))

    ## iterate in train and test, add the rgb dimention
    def add_rgb_dimention(array):
        scaler_map = cm.ScalarMappable(cmap="Oranges")
        array = scaler_map.to_rgba(array)[:, :-1]
        return array

    for i in range(num_samples):
        xtrain[i] = add_rgb_dimention(x_train[i])

    x_train = xtrain.reshape(num_samples, args.depth, args.height, args.width, args.in_channels)
    y_train = y_train.reshape(num_samples, 1)
    return x_train, y_train


def train(args: argparse.Namespace, model: nn.Module, optimizer: samba.optim.SGD) -> None:
    loss_thresh = 1.1
    if MOCK_SAMBA_RUNTIME:
        args.epochs = 1
        loss_thresh = float('inf')

    x_train, y_train = prepare_dataloader(args)

    # Train the model
    total_step = len(x_train)

    hyperparam_dict = {"lr": args.lr}
    pbar = tqdm(range(args.epochs))

    for epoch in pbar:
        avg_loss = 0
        for i, (images, labels) in enumerate(zip(x_train, y_train)):
            images = torch.tensor(images).permute(3, 0, 1, 2).unsqueeze(0).float()
            labels = torch.tensor(labels)
            sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
            sn_labels = samba.from_torch_tensor(labels, name='label', batch_dim=0)

            loss = samba.session.run(input_tensors=(sn_images, sn_labels),
                                     output_tensors=model.output_tensors,
                                     hyperparam_dict=hyperparam_dict)[0]
            loss = samba.to_torch(loss)
            avg_loss += loss.mean()
        pbar.set_description('epoch: {}, loss: {}'.format(epoch, avg_loss / total_step))

    final_loss = avg_loss / total_step
    if args.acc_test:
        assert final_loss < loss_thresh, f"Actual loss {final_loss} higher than expected {loss_thresh}"


def test(args: argparse.Namespace, model: nn.Module, inputs: Tuple[samba.SambaTensor],
         outputs: Tuple[samba.SambaTensor]) -> List[Tuple[str, samba.SambaTensor]]:
    model.zero_grad()
    inputs[0].grad = None

    gold_loss = model(*inputs)
    gold_loss.backward()

    samba_loss = samba.session.run(input_tensors=inputs, output_tensors=outputs)[0]

    # check that all samba and torch outputs match numerically
    print(f'samba output abs sum: {samba_loss.abs().sum().item()}')
    print(f'gold output abs sum: {gold_loss.abs().sum().item()}')

    utils.assert_close(samba_loss, gold_loss, f'output', threshold=0.001, visualize=args.visualize)
    print(f'gold_input_grad: {inputs[0].grad.abs().sum().item()}')
    print(f'samba_input_grad: {inputs[0].sn_grad.abs().sum().item()}')
    utils.assert_close(inputs[0].grad, inputs[0].sn_grad, f'input_grad', threshold=0.001, visualize=args.visualize)

    print(f'gold_conv_layer1: {model.conv_layer1[0].weight.grad.abs().sum().item()}')
    print(f'samba_conv_layer1: {model.conv_layer1[0].weight.sn_grad.abs().sum().item()}')
    utils.assert_close(model.conv_layer1[0].weight.grad,
                       model.conv_layer1[0].weight.sn_grad,
                       f'weight_grad1',
                       threshold=0.003,
                       visualize=args.visualize)

    print(f'gold_conv_layer2: {model.conv_layer2[0].weight.grad.abs().sum().item()}')
    print(f'samba_conv_layer2: {model.conv_layer2[0].weight.sn_grad.abs().sum().item()}')
    utils.assert_close(model.conv_layer2[0].weight.grad,
                       model.conv_layer2[0].weight.sn_grad,
                       f'weight_grad2',
                       threshold=0.003,
                       visualize=args.visualize)


def main():
    args = parse_app_args(common_parser_fn=add_args, dev_mode=True)
    utils.set_seed(256)

    model = Lenet3D()

    model.bfloat16().float()
    samba.from_torch_model_(model)

    metadata = None
    if args.enable_tiling:
        metadata = {
            ConvTilingMetadata.key:
            ConvTilingMetadata(original_size=[args.batch_size, args.in_channels, args.depth, args.height, args.width])
        }

    optim = samba.optim.SGD(model.parameters(), lr=args.lr) if not args.inference else None
    inputs = get_inputs(args)

    if args.command == "test":
        utils.trace_graph(model,
                          inputs,
                          optim,
                          pef=args.pef,
                          mapping=args.mapping,
                          init_output_grads=not args.inference)
        test(args, model, inputs, model.output_tensors)

    elif args.command == "run":
        utils.trace_graph(model,
                          inputs,
                          optim,
                          pef=args.pef,
                          mapping=args.mapping,
                          init_output_grads=not args.inference)
        train(args, model, optim)

    elif args.command == 'measure-performance':
        # Get inference latency and throughput statistics
        utils.trace_graph(model,
                          inputs,
                          optim,
                          pef=args.pef,
                          mapping=args.mapping,
                          init_output_grads=not args.inference)
        utils.measure_performance(model,
                                  inputs,
                                  args.batch_size,
                                  args.n_chips,
                                  args.inference,
                                  run_graph_only=args.run_graph_only,
                                  n_iterations=args.num_iterations,
                                  json=args.bench_report_json,
                                  compiled_stats_json=args.compiled_stats_json,
                                  data_parallel=args.data_parallel,
                                  reduce_on_rdu=args.reduce_on_rdu,
                                  min_duration=args.min_duration)
    else:
        common_app_driver(
            args,
            model,
            inputs,
            optim,
            "Conv3d",
            init_output_grads=not args.inference,
            squeeze_bs_dim=True,
            metadata=metadata,
        )


if __name__ == '__main__':
    main()
