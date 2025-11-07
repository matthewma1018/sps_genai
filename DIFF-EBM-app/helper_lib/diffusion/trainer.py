import torch
import torch.nn as nn

def _precompute_schedule(T=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)            # (T,)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)                                  # (T,)
    return betas, alphas, alpha_bar

def train_diffusion(
    model,
    data_loader,
    device="cuda",
    epochs=10,
    timesteps=1000,
    lr=1e-4,
):
    """
    diffusion training loop (DDPM-style noise prediction).
    - x0 is assumed in [0,1] (matches your minimal data loader).
    - Uniform timestep sampling, linear beta schedule.
    """
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    _, _, alpha_bar = _precompute_schedule(T=timesteps, device=device)  # (T,)

    for epoch in range(epochs):
        for x0, _ in data_loader:
            x0 = x0.to(device)  # (B,3,32,32), in [0,1]

            B = x0.size(0)
            t = torch.randint(low=0, high=timesteps, size=(B,), device=device)   # (B,)

            # Sample noise and construct x_t
            eps = torch.randn_like(x0)
            a_bar_t = alpha_bar.gather(0, t).view(B, 1, 1, 1)                    # (B,1,1,1)
            x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * eps

            # Predict noise
            eps_pred = model(x_t, t)

            loss = mse(eps_pred, eps)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch+1}/{epochs} - loss: {loss.item():.4f}")

    return model