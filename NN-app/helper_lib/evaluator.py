import torch

def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # We don't need to calculate gradients for evaluation, which saves memory and computation.
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: get model predictions
            outputs = model(images)

            # Calculate the loss for this batch
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate the number of correct predictions
            # torch.max returns (values, indices) of the max elements. We want the indices.
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = (correct_predictions / total_samples) * 100
    return avg_loss, accuracy