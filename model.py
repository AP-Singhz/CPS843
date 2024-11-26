import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations: resize, normalize, and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet-50 requires 224x224 images
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # Dataset directories
    train_dir = r'C:\Users\arora\Downloads\dataset\train'  # Update with your actual path
    test_dir = r'C:\Users\arora\Downloads\dataset\test'

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # Load pre-trained ResNet-50 model
    model = models.resnet50(weights="IMAGENET1K_V1")  # Use the recommended weights API

    model.fc = nn.Linear(model.fc.in_features, 2)  # Modify the final layer for binary classification
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 10  # Adjustable 
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Track the loss
            running_loss += loss.item()

            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print statistics for each epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # After training is complete, save the model
    torch.save(model.state_dict(), 'resnet50_trained_model.pth')
    print("Model saved successfully!")

    # Test the model
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

# Use the 'if __name__ == "__main__":' guard to handle multiprocessing correctly
if __name__ == "__main__":
    main()
