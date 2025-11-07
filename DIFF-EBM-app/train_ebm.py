import torch
from helper_lib.data_loader import get_data_loader
from helper_lib.ebm.model import get_model
from helper_lib.ebm.trainer import train_ebm
from helper_lib.utils import get_device, save_model

def main():
    device = get_device()
    loader = get_data_loader(data_dir="data", batch_size=128, train=True)

    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model = train_ebm(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        device=device,
        epochs=10,
    )

    save_model(model, model_type="ebm")

if __name__ == "__main__":
    main()