import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize images from [0, 1] to [-1, 1] to match the Generator's Tanh output range.
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,  # Shuffle the data only for the training set
    )

    return loader