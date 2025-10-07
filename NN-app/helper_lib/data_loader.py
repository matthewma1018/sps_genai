import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        # Assignment Part 1 specification
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR10(
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