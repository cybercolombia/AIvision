import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import sambaflow.samba.utils as utils
from sambaflow import samba
from sambaflow.samba.env import use_mock_samba_runtime
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.benchmark_acc import AccuracyReport
from sambaflow.samba.utils.pef_utils import get_pefmeta

MOCK_SAMBA_RUNTIME = use_mock_samba_runtime()


class LeNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=(1, 1),
                               bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=(1, 1),
                               bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        x = self.conv1(inputs).relu()
        x = self.maxpool1(x)
        x = self.conv2(x).relu()
        x = self.maxpool2(x)
        x = torch.reshape(x, [x.shape[0], -1])
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        out = self.fc3(x)
        loss = self.criterion(out, labels)
        return loss, out


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--num-classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight-decay', type=float, default=0.01)


def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument('--train-torch', action='store_true', help='train FP32 PyTorch version of application')
    parser.add_argument('-n', '--num-iterations', type=int, default=100, help='Number of iterations to run the pef for')
    parser.add_argument('-e', '--num-epochs', type=int, default=1)
    parser.add_argument('--log-path', type=str, default='checkpoints')
    parser.add_argument('--dump-interval', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--measure-train-performance', action='store_true')
    parser.add_argument('--acc-test', action='store_true', help='Option for accuracy guard test in RDU regression.')
    parser.add_argument('--data-dir',
                        type=str,
                        default='mnist_data',
                        help="The folder to download the MNIST dataset to.")


def get_inputs(args: argparse.Namespace) -> Tuple[samba.SambaTensor, samba.SambaTensor]:
    images = samba.randn(args.batch_size, 1, 28, 28, name='image', batch_dim=0)
    labels = samba.randint(args.num_classes, (args.batch_size, ), name='label', batch_dim=0)
    return (images, labels)


def run_test(args: argparse.Namespace, model: nn.Module, inputs: Tuple[samba.SambaTensor],
             outputs: Tuple[samba.SambaTensor]) -> None:
    loss_gold, output_gold = model(*inputs)
    loss_samba, output_samba = samba.session.run(input_tensors=inputs, output_tensors=outputs)

    # check that all samba and torch outputs match numerically
    print(f'gold_output_abs_sum: {output_gold.abs().sum()}', output_gold)
    print(f'samba_output_abs_sum: {output_samba.abs().sum()}', output_samba)
    utils.assert_close(output_samba, output_gold, f'forward output', threshold=0.085, visualize=args.visualize)

    print(f'gold_loss_abs_sum: {loss_gold.abs().sum()}', loss_gold)
    print(f'samba_loss_abs_sum: {loss_samba.abs().sum()}', loss_samba)
    utils.assert_close(loss_samba, loss_gold, f'forward output', threshold=0.032, visualize=args.visualize)

    if not args.inference:
        loss_gold.mean().backward()
        conv1_grad_gold = model.conv1.weight.grad
        conv1_grad_samba = model.conv1.weight.sn_grad
        utils.assert_close(conv1_grad_samba,
                           conv1_grad_gold,
                           'conv1__weight__grad',
                           threshold=3e-2,
                           visualize=args.visualize)

        conv2_grad_gold = model.conv2.weight.grad
        conv2_grad_samba = model.conv2.weight.sn_grad
        utils.assert_close(conv2_grad_samba,
                           conv2_grad_gold,
                           'conv2__weight__grad',
                           threshold=7e-2,
                           visualize=args.visualize)

        fc1_grad_gold = model.fc1.weight.grad
        fc1_grad_samba = model.fc1.weight.sn_grad
        utils.assert_close(fc1_grad_samba, fc1_grad_gold, 'fc1__weight__grad', threshold=3e-2, visualize=args.visualize)

        fc2_grad_gold = model.fc2.weight.grad
        fc2_grad_samba = model.fc2.weight.sn_grad
        utils.assert_close(fc2_grad_samba, fc2_grad_gold, 'fc2__weight__grad', threshold=3e-3, visualize=args.visualize)

        fc3_grad_gold = model.fc3.weight.grad
        fc3_grad_samba = model.fc3.weight.sn_grad
        utils.assert_close(fc3_grad_samba, fc3_grad_gold, 'fc3__weight__grad', threshold=3e-3, visualize=args.visualize)


def prepare_dataloader(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        # norm by mean and var
        transforms.Normalize((0.1307, ), (0.3081, )),
        # Reshape image to 1x28x28
        lambda x: x.reshape((1, 28, 28)),
    ])

    # Train the model on RDU using the MNIST dataset (images and labels)
    train_dataset = torchvision.datasets.MNIST(root=f'{args.data_dir}', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=f'{args.data_dir}', train=False, transform=transform)

    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader


def train(args: argparse.Namespace, model: nn.Module, optimizer: samba.optim.SGD) -> None:
    train_loader, test_loader = prepare_dataloader(args)
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
                                              hyperparam_dict=hyperparam_dict,
                                              data_parallel=args.data_parallel,
                                              reduce_on_rdu=args.reduce_on_rdu)
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            avg_loss += loss.mean()

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
            min_bound = 95.0 if not MOCK_SAMBA_RUNTIME else -1.0
            max_bound = 97.0 if not MOCK_SAMBA_RUNTIME else 100.0
            assert args.num_epochs == 1, "Accuracy test only supported for 1 epoch"
            assert test_acc > min_bound and test_acc < max_bound, "Test accuracy not within specified bounds."

    if args.acc_report_json is not None:
        val_metrics = {'acc': test_acc.item() / 100.0, 'loss': total_loss.item() / len(test_loader)}
        report = AccuracyReport(val_metrics=val_metrics,
                                batch_size=args.batch_size,
                                num_iterations=args.num_epochs * total_step)
        report.save(args.acc_report_json)


def main():
    args = parse_app_args(dev_mode=True,
                          common_parser_fn=add_common_args,
                          test_parser_fn=add_run_args,
                          run_parser_fn=add_run_args)
    utils.set_seed(256)
    model = LeNet(args.num_classes)
    samba.from_torch_model_(model)

    inputs = get_inputs(args)

    optimizer = samba.optim.SGD(model.parameters(), lr=0.0) if not args.inference else None
    if args.command == "compile":
        samba.session.compile(model,
                              inputs,
                              optimizer,
                              name='lenet',
                              squeeze_bs_dim=True,
                              config_dict=vars(args),
                              pef_metadata=get_pefmeta(args, model))

    elif args.command == "test":
        #Test Lenet
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        outputs = model.output_tensors
        run_test(args, model, inputs, outputs)
    elif args.command == "run":
        #Run Lenet
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        train(args, model, optimizer)


if __name__ == '__main__':
    main()
