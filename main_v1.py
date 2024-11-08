import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import barcode_Dataset
from train import train_model
from model import SimpleCNN

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Dataset and Data Loaders
root_dir_path = os.path.join('C:', 'Users', 'venky', 'Downloads', 'archive', 'images')
dataset = barcode_Dataset(root_dir=root_dir_path, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize Model, Criterion and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
