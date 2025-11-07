train_ebm.py:
Trains the minimal EBM on CIFAR-10 using a simple NCE setup (real vs. uniform noise).
Saves weights to checkpoints/ebm.pt.

train_diff.py:
Trains a minimal UNet to predict noise (DDPM objective).
Saves weights to checkpoints/diffusion.pt.

main.py:
Loads checkpoints and serves the API.

Endpoints:

GET /ebm/generate → Langevin samples from the EBM; returns PNG (grid if N>1).

GET /diffusion/generate → DDPM samples from the diffusion model; returns PNG.