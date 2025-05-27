import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
from typing import Tuple

from model import TestModel

DATA_PATH = '/home/carlos/Documents/CyberComputingConsulting/AIvision/repo/AIvision/sambanova/conversion/logclass/data/'

class CustomDataset(Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        self.data, self.targets = self._read_data()

    def _read_data(self):
        data = []
        targets = []
        with open(self.data_file, 'r') as f:
            for line in f:
                row = [float(x) for x in line.strip().split(',')]
                data.append(row[:2])
                targets.append(row[2])
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_data = torch.tensor(self.data[idx], dtype=torch.float32)
        sample_target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sample_data, sample_target

def prepare_dataloader(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Data set
    train_dataset = CustomDataset(DATA_PATH + "train.txt")
    test_dataset = CustomDataset(DATA_PATH + "test.txt")
    
    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader

def train(model: nn.Module, device: str, train_loader: DataLoader, 
         num_epochs: int, learning_rate: float) -> Tuple[list[float], list[float]]:
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #Stochastic gradient descent

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        model.train(True)
        for i, data in enumerate(train_loader):
            points, labels = data
            points, labels = points.to(device), labels.to(device) 
            
            # Run the forward pass
            loss, outputs = model(points, labels)
            loss_list.append(loss.item())
            
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
            del points
            del labels
            torch.cuda.empty_cache()
    
    return loss_list, acc_list

def main():
    # Hyperparameters
    num_epochs = 50
    batch_size = 100
    learning_rate = 0.001
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    train_loader, test_loader = prepare_dataloader(batch_size)
    
    model = TestModel().to(device)

    # Train the model
    loss_list, acc_list = train(model, device, train_loader, num_epochs, learning_rate)


if __name__ == '__main__':
    main()