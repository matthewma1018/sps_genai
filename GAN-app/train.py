import torch
import torch.nn as nn
import torch.optim as optim
from helper_lib.data_loader import get_data_loader
from helper_lib.model import Generator, Discriminator
from helper_lib.trainer import train_gan
from helper_lib.utils import get_device, save_model

DATA_DIR = "./data/mnist"
BATCH_SIZE = 64
LEARNING_RATE = 0.0002  # A common learning rate for GANs
EPOCHS = 500
Z_DIM = 100  # Noise dimension from spec

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_loader = get_data_loader(DATA_DIR, batch_size=BATCH_SIZE, train=True)

    gen = Generator(z_dim=Z_DIM).to(device)
    disc = Discriminator().to(device)

    # Initialize Loss and Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Use BCEWithLogitsLoss because Discriminator doesn't have a sigmoid
    criterion = nn.BCEWithLogitsLoss()

    print("Starting GAN training...")
    trained_gen = train_gan(
        gen,
        disc,
        train_loader,
        criterion,
        opt_gen,
        opt_disc,
        device,
        epochs=EPOCHS,
        z_dim=Z_DIM
    )

    print("Training complete.")

    save_model(trained_gen, "gan_generator.pth")

if __name__ == "__main__":
    main()