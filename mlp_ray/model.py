"""Shared model + data-loading code for the FashionMNIST MLP.

Factored out so train_ray_tune.py and classify.py always train/load the exact
same model.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Always use a local ./data folder, independent of any dataset that may exist
# elsewhere on the machine. Downloaded automatically on first run.
DATA_DIR = os.path.join(THIS_DIR, "data")


class MLP(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_data(batch_size, data_dir=DATA_DIR):
    """Downloads FashionMNIST on first use (~30MB) and returns loaders + class names."""
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    valid_data = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, train_data.classes


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)
