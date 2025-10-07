import torch

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, data_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return model