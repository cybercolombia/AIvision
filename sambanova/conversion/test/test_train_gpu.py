import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
from typing import Tuple

from model import TestModel

DATA_PATH = '/home/carlos/Documents/CyberComputingConsulting/AIvision/repo/AIvision/sambanova/conversion/test/data/'

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

def prepare_dataloader(batch_size: int) -> Tuple[DataLoader]:
    # Data set
    train_dataset = CustomDataset(DATA_PATH + "X.npy", DATA_PATH + "y.npy")
    print("train_dataset size:", len(train_dataset))
    print("train_dataset data shape:", train_dataset.data.shape)
    print("train_dataset targets shape:", train_dataset.targets.shape)
    # Check if the dataset is empty
    if len(train_dataset) == 0:
        raise ValueError("The dataset is empty. Please check the data files.")
    
    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader

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
            loss, outputs = model(points)
            loss_list.append(loss.item())
            
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
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

    train_loader = prepare_dataloader(batch_size)
    
    model = TestModel().to(device)

    # Train the model
    loss_list, acc_list = train(model, device, train_loader, num_epochs, learning_rate)


if __name__ == '__main__':
    main()