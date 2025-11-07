from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.ToTensor()  # images in [0,1], 3 channels
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
    )