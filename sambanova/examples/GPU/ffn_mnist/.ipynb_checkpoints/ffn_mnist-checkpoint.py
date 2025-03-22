# Adaptation of Sambanova example to GPU

import argparse
import os
import queue
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
#from modelbox import (BatchPredictStatus, Metric, OnlinePredictStatus, PredictionDataset, StatusSender, TrainStatus,
#                      pack_value, write_prediction_results)
#from modelbox.modelbox_api import ModelboxInterface
from termcolor import colored
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

# import sambaflow.samba.utils as utils
# import sambaflow.samba.utils.dataset.mnist as mnist
# from sambaflow import samba
# from sambaflow.samba.utils.argparser import parse_app_args
# from sambaflow.samba.utils.pef_utils import get_pefmeta_dict



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

    def _get_model(self, device):
        params = self.params
        model = FFNLogReg(params['num_features'], params['ffn_dim_1'], params['ffn_dim_2'], params['num_classes'])
        model = model.to(device)
        return model