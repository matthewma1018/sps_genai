import torch
import torch.nn as nn
import torchvision
import os

save_dir = "generated_images"
os.makedirs(save_dir, exist_ok=True)

def train_gan(gen, disc, data_loader, criterion, opt_gen, opt_disc, device, epochs, z_dim):
    # Set models to training mode
    gen.train()
    disc.train()

    for epoch in range(epochs):
        # track the average loss
        total_loss_disc = 0.0
        total_loss_gen = 0.0

        for batch_idx, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]

            # The discriminator learns to maximize:
            # log(D(real_images)) + log(1 - D(G(noise)))

            # Zero the discriminator's gradients
            opt_disc.zero_grad()

            # Pass real images through the discriminator
            disc_real_pred = disc(real_images)

            # Calculate loss for real images
            # We want the discriminator to output 1s ("real")
            real_labels = torch.ones_like(disc_real_pred, device=device)
            loss_disc_real = criterion(disc_real_pred, real_labels)

            # Generate a batch of noise
            noise = torch.randn(batch_size, z_dim, device=device)

            # Generate fake images from noise
            # .detach() stops gradients from flowing back to the generator
            fake_images = gen(noise).detach()

            # Pass fake images through the discriminator
            disc_fake_pred = disc(fake_images)

            # Calculate loss for fake images
            # We want the discriminator to output 0s ("fake")
            fake_labels = torch.zeros_like(disc_fake_pred, device=device)
            loss_disc_fake = criterion(disc_fake_pred, fake_labels)

            # Total discriminator loss is the sum of real and fake losses
            loss_disc = loss_disc_real + loss_disc_fake
            loss_disc.backward()
            opt_disc.step()

            total_loss_disc += loss_disc.item()

            # The generator learns to minimize:
            # log(1 - D(G(noise)))
            # or maximize:
            # log(D(G(noise))) (this is the non-saturating loss, often better)

            # Zero the generator's gradients
            opt_gen.zero_grad()

            # Generate a new batch of fake images
            # We must NOT use .detach() here, we need the gradients
            noise = torch.randn(batch_size, z_dim, device=device)
            fake_images_for_gen = gen(noise)

            # Pass the new fake images through the discriminator
            disc_pred_for_gen = disc(fake_images_for_gen)

            # Calculate generator's loss
            # The generator's goal is to make the discriminator output 1s ("real").
            generator_target_labels = torch.ones_like(disc_pred_for_gen, device=device)
            loss_gen = criterion(disc_pred_for_gen, generator_target_labels)

            # --- Update Generator ---
            loss_gen.backward()
            opt_gen.step()

            total_loss_gen += loss_gen.item()

        avg_loss_disc = total_loss_disc / len(data_loader)
        avg_loss_gen = total_loss_gen / len(data_loader)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss D: {avg_loss_disc:.4f}, Loss G: {avg_loss_gen:.4f}")

        # Save a grid of images
        # with torch.no_grad():
        #     # Use a fixed noise vector to inspect the images evolve over time
        #     if 'fixed_noise' not in locals():
        #         fixed_noise = torch.randn(64, z_dim, device=device)
        #
        #     sample_images = gen(fixed_noise).cpu()
        #
        #     torchvision.utils.save_image(
        #         sample_images,
        #         f"{save_dir}/epoch_{epoch + 1:03d}.png",
        #         normalize=True
        #     )
    return gen