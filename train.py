import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Move model to device
    model.train()

    for epoch in range(epochs):
        train_error = 0

        # Training phase
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()  # Ensure labels are float tensors for criterion

            outputs = model(images)  # Forward pass
            
            # Ensure outputs and labels have the same shape
            if outputs.shape != labels.shape:
                outputs = outputs.view_as(labels)

            error = criterion(outputs, labels)  # Compute loss
            
            optimizer.zero_grad()  # Reset gradients
            error.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            train_error += error.item()

        avg_train_error = train_error / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Error: {avg_train_error:.4f}")

        # Validation phase
        model.eval()
        val_error = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                
                outputs = model(images)
                
                # Ensure shapes match
                if outputs.shape != labels.shape:
                    outputs = outputs.view_as(labels)

                error = criterion(outputs, labels)
                val_error += error.item()
        
        avg_val_error = val_error / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Error: {avg_val_error:.4f}")

    print("Training complete.")
