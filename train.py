import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    for epoch in range(epochs):
        train_error = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            
            outputs = model(images)
            error = criterion(outputs.squeeze(), labels)
            
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            
            train_error += error.item()
        
        avg_train_error = train_error/len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Error: {avg_train_error:.4f}")
        
        model.eval()
        val_error = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                error = criterion(outputs.squeeze(), labels)
                val_error += error.item()
            
        avg_val_error = val_error/len(val_loader)
        print(f"Validation Error: {avg_val_error:.4f}")
