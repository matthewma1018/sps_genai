import torch
from helper_lib.data_loader import get_data_loader
from helper_lib.diffusion.model import get_model
from helper_lib.diffusion.trainer import train_diffusion
from helper_lib.utils import get_device, save_model

def main():
    device = get_device()
    print(device)
    loader = get_data_loader(data_dir="data", batch_size=128, train=True)

    model = get_model("DIFF").to(device)
    model = train_diffusion(
        model=model,
        data_loader=loader,
        device=device,
        epochs=100,
        timesteps=1000,   # must match sampler later
        lr=1e-4,
    )

    # Save to checkpoints/diffusion.pt
    save_model(model, model_type="diffusion")

if __name__ == "__main__":
    main()