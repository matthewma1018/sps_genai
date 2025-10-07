from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.utils import get_device, save_model

import torch.nn as nn
import torch.optim as optim

DATA_DIR = "./data"  # A single directory for the CIFAR10 dataset
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

def main():
    device = get_device()
    print(f"Using device: {device}")

    train_loader = get_data_loader(DATA_DIR, batch_size=BATCH_SIZE, train=True)
    test_loader = get_data_loader(DATA_DIR, batch_size=BATCH_SIZE, train=False)

    # The model is created but not yet moved to the device; train_model will handle that.
    model = get_model("CNN")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trained_model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)

    test_loss, test_accuracy = evaluate_model(trained_model, test_loader, criterion, device=device)

    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.2f}%")

    save_model(trained_model, "cifar10_cnn.pth")

if __name__ == "__main__":
    main()
