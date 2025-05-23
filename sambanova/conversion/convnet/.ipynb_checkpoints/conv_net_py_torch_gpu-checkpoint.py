import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
from typing import Tuple

# Convolutional neural network (two convolutional layers)
from model import ConvNet

DATA_PATH = '/home/carlos/Documents/CyberComputingConsulting/AIvision/sambanova/conversion/convnet/MNISTData'
MODEL_STORE_PATH = '/home/carlos/Documents/CyberComputingConsulting/AIvision/sambanova/conversion/convnet/pytorch_models'


def prepare_dataloader(batch_size: int) -> Tuple[DataLoader, DataLoader]:    
    # transforms to apply to the data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
    
    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model: nn.Module, device: str, train_loader: DataLoader, 
         num_epochs: int, learning_rate: float) -> Tuple[list[float], list[float]]:
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        model.train(True)
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device) 
            
            # Run the forward pass
            loss, outputs = model(images, labels)
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
            
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
            del images
            del labels
            torch.cuda.empty_cache()
    
    return loss_list, acc_list

def main():
    # Hyperparameters
    num_epochs = 6
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    train_loader, test_loader = prepare_dataloader(batch_size)
    
    model = ConvNet().to(device)

    # Train the model
    loss_list, acc_list = train(model, device, train_loader, num_epochs, learning_rate)

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) 
            _, outputs = model(images, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    # Save the model and plot
    torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')
    
    p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
    p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
    p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
    p.line(np.arange(len(loss_list)), loss_list)
    p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
    show(p)

if __name__ == '__main__':
    main()