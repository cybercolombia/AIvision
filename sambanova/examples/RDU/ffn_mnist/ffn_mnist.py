# Copyright © 2020-2024 by SambaNova Systems, Inc. Disclosure, reproduction,
# reverse engineering, or any other use made without the advance written
# permission of SambaNova Systems, Inc. is unauthorized and strictly
# prohibited. All rights of ownership and enforcement are reserved.

import argparse
import os
import queue
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from modelbox import (BatchPredictStatus, Metric, OnlinePredictStatus, PredictionDataset, StatusSender, TrainStatus,
                      pack_value, write_prediction_results)
from modelbox.modelbox_api import ModelboxInterface
from termcolor import colored
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

import sambaflow.samba.utils as utils
import sambaflow.samba.utils.dataset.mnist as mnist
from sambaflow import samba
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.pef_utils import get_pefmeta_dict


class FFN(nn.Module):
    """Feed Forward Network"""
    def __init__(self, num_features: int, ffn_dim_1: int, ffn_dim_2: int) -> None:
        super().__init__()
        self.gemm1 = nn.Linear(num_features, ffn_dim_1, bias=False)
        self.relu = nn.ReLU()
        self.gemm2 = nn.Linear(ffn_dim_1, ffn_dim_2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gemm1(x)
        out = self.relu(out)
        out = self.gemm2(out)
        return out


class LogReg(nn.Module):
    """Logreg"""
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.lin_layer = nn.Linear(in_features=num_features, out_features=num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        out = self.lin_layer(inputs)
        loss = self.criterion(out, targets)
        return loss, out


class FFNLogReg(nn.Module):
    """Feed Forward Network + LogReg"""
    def __init__(self, num_features: int, ffn_embedding_size: int, embedding_size: int, num_classes: int) -> None:
        super().__init__()
        self.ffn = FFN(num_features, ffn_embedding_size, embedding_size)
        self.logreg = LogReg(embedding_size, num_classes)
        self._init_params()

    def _init_params(self) -> None:
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        out = self.ffn(inputs)
        loss, out = self.logreg(out, targets)
        return loss, out


"""
Model refactored for Modelbox interface.
"""


class FFNLogRegModelBox(ModelboxInterface):
    params = {"arch": '2.0', "inference": False, "mapping": "section"}

    def __init__(self, params: Dict[str, Any]):
        super(FFNLogRegModelBox, self).__init__(params, add_args_fn=add_args)
        print(f"Initialized FFNLogReg model with params {self.params}")

    def _get_model(self):
        params = self.params
        model = FFNLogReg(params['num_features'], params['ffn_dim_1'], params['ffn_dim_2'], params['num_classes'])
        samba.from_torch_model_(model)
        return model

    def _get_optimizer(self, model):
        params = self.params
        if params['inference']:
            optimizer = None
        else:
            optimizer = samba.optim.SGD(model.parameters(),
                                        lr=params['lr'],
                                        momentum=params['momentum'],
                                        weight_decay=params['weight_decay'])
        return optimizer

    def _get_inputs(self):
        params = self.params
        ipt = samba.randn(params['batch_size'], params['num_features'], name='image', batch_dim=0).bfloat16().float()
        tgt = samba.randint(params['num_classes'], (params['batch_size'], ), name='label', batch_dim=0)
        return (ipt, tgt)

    def compile(self, output: queue.Queue = None) -> None:
        params = self.params

        model = self._get_model()
        inputs = self._get_inputs()
        optimizer = self._get_optimizer(model)

        # Run model analysis and compile, this step will produce a PEF.
        status_sender = StatusSender(output)
        status_sender.compile_and_track_progress(samba.session.compile,
                                                 params['log_dir'],
                                                 model,
                                                 inputs,
                                                 optimizer,
                                                 name='ffn_mnist_torch',
                                                 config_dict=params,
                                                 pef_metadata=get_pefmeta_dict(params, model))

    def preprocess(self, data_dir: str):
        params = self.params

        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            # norm by mean and var
            transforms.Normalize((0.1307, ), (0.3081, )),
            # pad to 800 num of features
            lambda x: torch.cat([x.view(-1), torch.zeros(params['num_features'] - 784)]),
        ])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=dataset_transform)
        eval_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=dataset_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=dataset_transform)

        return train_dataset, eval_dataset, test_dataset

    def _make_metrics(self, epoch=None, cur_loss=None, test_acc=None):
        metrics = []
        if epoch:
            metrics.append(Metric(name="epoch", value=epoch))
        if cur_loss:
            metrics.append(Metric(name="loss", value=cur_loss))
        if test_acc:
            metrics.append(Metric(name="test_acc", value=test_acc))
        return metrics

    def _load_checkpoint(self, model, optimizer, init_ckpt_path: str):
        print(f"Loading checkpoint from file {init_ckpt_path}")
        ckpt = torch.load(init_ckpt_path)
        ckpt['global_step']
        if model:
            model.load_state_dict(ckpt['model'])
        if optimizer:
            optimizer.load_state_dict(ckpt['optimizer'])

    def _save_checkpoint(self, global_step, model, optimizer, ckpt_dir):
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)

        state_dict = {'global_step': global_step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        ckpt_path = f"{ckpt_dir}/{global_step}.pt"
        torch.save(state_dict, ckpt_path)
        return ckpt_path

    def train(self, data_dir: str, ckpt_dir: str, statuses: queue.Queue = None):
        params = self.params
        if ckpt_dir:
            params['ckpt_dir'] = ckpt_dir

        train_dataset, eval_dataset, _ = self.preprocess(data_dir)

        # Dataset needs implement prepare_dataloader()
        train_loader, eval_loader = mnist.prepare_dataloader(params, train_dataset, eval_dataset)

        model = self.model
        optimizer = self.optimizer

        if not self.run_cpu:
            hyperparam_dict = {
                "lr": params['lr'],
                "momentum": params['momentum'],
                "weight_decay": params['weight_decay']
            }

        global_step = 0

        total_step = len(train_loader)
        status_sender = StatusSender(statuses) if statuses is not None else None
        for epoch in range(params['num_epochs']):
            if status_sender is not None:
                status_sender.send_train_or_batch_infer_progress("train",
                                                                 completed_steps=epoch,
                                                                 total_steps=params['num_epochs'])
            # Run training
            avg_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                if self.run_cpu:
                    loss, outputs = model(images, labels)
                    loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
                    sn_labels = samba.from_torch_tensor(labels, name='label', batch_dim=0)

                    loss, outputs = samba.session.run(input_tensors=[sn_images, sn_labels],
                                                      output_tensors=model.output_tensors,
                                                      hyperparam_dict=hyperparam_dict,
                                                      data_parallel=params['data_parallel'],
                                                      reduce_on_rdu=params['reduce_on_rdu'])
                    loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)

                avg_loss += loss.mean()
                global_step += 1

                if (i + 1) % 10000 == 0:
                    if status_sender is not None:
                        status_sender.send_train_or_batch_infer_progress("train",
                                                                         completed_steps=(epoch + (i + 1) / total_step),
                                                                         total_steps=params['num_epochs'])
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, params['num_epochs'], i + 1,
                                                                             total_step, avg_loss / (i + 1)))

            # Compute validation
            if not self.run_cpu:
                samba.session.to_cpu(model)

            eval_acc = 0.0
            cur_loss = 0.0
            with torch.no_grad():
                correct = 0
                total = 0
                total_loss = 0
                for images, labels in eval_loader:
                    loss, outputs = model(images, labels)
                    total_loss += loss.mean()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

                eval_acc = 100.0 * correct / total
                cur_loss = total_loss.item() / len(eval_loader)
                print('Test Accuracy: {:.2f}'.format(eval_acc), ' Loss: {:.4f}'.format(cur_loss))

            # Save status
            if statuses is not None:
                metrics = self._make_metrics(epoch, cur_loss, eval_acc)
                status = TrainStatus(metrics=metrics, ckpt_path=None, complete=False)
                statuses.put(status)

        total_loss.item() / (len(eval_loader))
        ckpt_path = self._save_checkpoint(global_step, model, optimizer, params['ckpt_dir'])

        if params.get('acc_test', False):
            assert params['num_epochs'] == 1, "Accuracy eval only supported for 1 epoch"
            assert eval_acc > 91.0 and eval_acc < 92.0, "Test accuracy not within specified bounds."

        if statuses is not None:
            statuses.put(
                TrainStatus(metrics=self._make_metrics(epoch, cur_loss, eval_acc),
                            ckpt_path=ckpt_path,
                            progress=1,
                            complete=True))

    def _make_predictions(self, predictions, labels_map):
        # Final predictions are the actual predicted classes and can be mapped to actual predicted labels
        if labels_map:
            final_prediction = [labels_map[pred] for pred in torch.max(predictions, axis=1).values.tolist()]
        else:
            final_prediction = [str(pred) for pred in torch.max(predictions, axis=1).values.tolist()]
        return final_prediction

    def _get_labels_map(self, dataset) -> Optional[Dict[int, str]]:
        """
        Retrieve labels map to connect predictions to actual meaningful labels
        """
        #TODO(andyc): add labels_map to datasets
        if hasattr(dataset, '_labels_map'):
            return dataset._labels_map
        else:
            return None

    def prepare(self, init_ckpt_path: str):
        """
        Prepares the model for inference through a future call to predict()
        """
        params = self.params

        # TODO: Do we do this during prepare or every call to predict?
        self.run_cpu = params['cpu']

        if not self.run_cpu:
            samba.session.reset()

        # Prepare the model for prediction
        self.model = self._get_model()
        self.optimizer = self._get_optimizer(self.model)

        if not self.run_cpu:
            inputs = self._get_inputs()
            utils.trace_graph(self.model, inputs, self.optimizer, pef=params['pef'], mapping=params['mapping'])

        if init_ckpt_path:
            self._load_checkpoint(self.model, self.optimizer, init_ckpt_path)
        else:
            if self.params["modelbox_mode"] != "train":
                print(colored('[WARNING]No valid initial checkpoint has been provided', 'red'))

    def online_predict(self, instances: List["modelbox_pb2.Value"], params: Dict[str, Any], results: queue.Queue):
        """
        Executes an online prediction, writing one Value to the results queue
        for each Value in the requests list.
        """
        params = self.params
        #TODO: remove conversion to PredictionDataset or collapse it with streaming dataset classes
        dataset = PredictionDataset(instances)

        loader = DataLoader(dataset,
                            batch_size=params.get('batch_size', 1),
                            drop_last=False,
                            shuffle=False,
                            num_workers=params.get('num_workers', 0))

        labels_map = self._get_labels_map(dataset)

        prediction_list = []

        for _, (images, labels) in enumerate(loader):
            if self.run_cpu:
                _, predictions = self.model(images, labels)
                predictions = samba.to_torch(predictions)
            else:
                sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
                sn_labels = samba.from_torch_tensor(labels, name='label', batch_dim=0)

                _, predictions = samba.session.run(input_tensors=[sn_images, sn_labels],
                                                   output_tensors=self.model.output_tensors,
                                                   section_types=['fwd'])
                predictions = samba.to_torch(predictions)

            prediction_list.extend(self._make_predictions(predictions.data, labels_map))
        prediction_list = [pack_value(prediction) for prediction in prediction_list]
        results.put(OnlinePredictStatus(prediction_list, complete=True))

    def batch_predict(self, data_dir: str, result_dir: str, statuses: queue.Queue):
        """
        Execute a batch prediction, reading from data_dir and writing to result_dir.
        Progress reports can be passed to the statuses queue as well as the final
        completion event.
        """
        params = self.params

        _, _, dataset = self.preprocess(data_dir)

        loader = DataLoader(dataset,
                            batch_size=params.get('batch_size', 1),
                            drop_last=False,
                            shuffle=False,
                            num_workers=params.get('num_workers', 0))

        labels_map = self._get_labels_map(dataset)
        avg_acc = 0.
        avg_loss = 0.
        total_batches = len(loader)
        status_sender = StatusSender(statuses) if statuses is not None else None
        for i, (images, labels) in enumerate(loader):

            # Send progress at start of loop so we don't duplicate progress=1 status
            if status_sender is not None:
                status_sender.send_train_or_batch_infer_progress("infer", completed_steps=i, total_steps=total_batches)

            if self.run_cpu:
                loss, predictions = self.model(images, labels)
                loss, predictions = samba.to_torch(loss), samba.to_torch(predictions)
            else:
                sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
                sn_labels = samba.from_torch_tensor(labels, name='label', batch_dim=0)

                loss, predictions = samba.session.run(input_tensors=[sn_images, sn_labels],
                                                      output_tensors=self.model.output_tensors,
                                                      section_types=['fwd'])
                loss, predictions = samba.to_torch(loss), samba.to_torch(predictions)

            step_loss = loss.mean()
            _, prediction_values = torch.max(predictions, axis=1)
            step_acc = 100.0 * (prediction_values == labels).sum() / labels.size(0)
            avg_loss += step_loss
            avg_acc += step_acc
            metrics = self._make_metrics(cur_loss=avg_loss / (i + 1), test_acc=avg_acc / (i + 1))

            if statuses is not None:
                prediction_list = self._make_predictions(predictions.data, labels_map)
                result_path = write_prediction_results(prediction_list, result_dir, i + 1)
                result = BatchPredictStatus(metrics=metrics,
                                            result_path=[result_path],
                                            progress=i / total_batches,
                                            complete=False)
                statuses.put(result)

        if statuses is not None:
            metrics = self._make_metrics(cur_loss=avg_loss / (i + 1), test_acc=avg_acc / (i + 1))
            result = BatchPredictStatus(metrics=metrics, result_path=[result_path], progress=1, complete=True)
            statuses.put(result)

    def test(self) -> None:
        params = self.params

        model = self._get_model()
        inputs = self._get_inputs()
        optimizer = self._get_optimizer(model)

        utils.trace_graph(model, inputs, optimizer, pef=params['pef'], mapping=params['mapping'])
        outputs = model.output_tensors
        outputs_gold = model(*inputs)

        outputs_samba = samba.session.run(input_tensors=inputs,
                                          output_tensors=outputs,
                                          data_parallel=params['data_parallel'],
                                          reduce_on_rdu=params['reduce_on_rdu'])

        # check that all samba and torch outputs match numerically
        for i, (output_samba, output_gold) in enumerate(zip(outputs_samba, outputs_gold)):
            print('samba:', output_samba)
            print('gold:', output_gold)
            utils.assert_close(output_samba, output_gold, f'forward output #{i}', threshold=1e-2)

        if not params['inference']:
            # training mode, check two of the gradients
            torch_loss, torch_gemm_out = outputs_gold
            torch_loss.mean().backward()

            # we choose two gradients from different places to test numerically
            gemm1_grad_gold = model.ffn.gemm1.weight.grad
            gemm1_grad_samba = model.ffn.gemm1.weight.sn_grad

            utils.assert_close(gemm1_grad_gold, gemm1_grad_samba, 'ffn__gemm1__weight__grad', threshold=1e-2)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--momentum', type=float, default=0.0, help="Momentum value for training")
    parser.add_argument('--weight-decay', type=float, default=1e-4, help="Weight decay for training")
    parser.add_argument('-e', '--num-epochs', type=int, default=1)
    parser.add_argument('--num-features', type=int, default=784)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--acc-test', action='store_true', help='Option for accuracy guard test in RDU regression.')
    parser.add_argument('--ffn-dim-1', type=int, default=32)
    parser.add_argument('--ffn-dim-2', type=int, default=32)
    parser.add_argument('--cpu', action='store_true', help='Run on CPU')
    parser.add_argument('--init-ckpt-path', type=str, default='', help='Path to load checkpoint')
    parser.add_argument('--ckpt-dir', type=str, default=os.getcwd(), help='Path to save checkpoint')


def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--num-workers', type=int, default=0)


def main(argv):
    """
    In order to guarantee that the modelbox app and the model when run directly have identical behavior, the main
    function should not do anything except parse arguments and call the modelbox APIs. Any other functionality here
    will not get run when invoking the model through modelbox, and is a red flag.
    """
    args = parse_app_args(argv=argv, common_parser_fn=add_args, run_parser_fn=add_run_args)
    params = vars(args)
    params['modelbox_mode'] = args.command
    modelbox = FFNLogRegModelBox(params)

    if args.command == "compile":
        # Run model analysis and compile, this step will produce a PEF.
        modelbox.compile()
    elif args.command == "test":
        modelbox.test()
    elif args.command == "run":
        if not args.inference:
            modelbox.prepare(params['init_ckpt_path'])
            modelbox.train(params['data_dir'], '', queue.Queue())
        else:
            modelbox.prepare(params['init_ckpt_path'])
            modelbox.batch_predict(params['data_dir'], '.', queue.Queue())


if __name__ == '__main__':
    main(sys.argv[1:])
