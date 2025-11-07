import torch

def _precompute_schedule(T=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)       # (T,)
    alphas = 1.0 - betas                                                 # (T,)
    alpha_bar = torch.cumprod(alphas, dim=0)                             # (T,)
    return betas, alphas, alpha_bar

@torch.no_grad()
def generate_samples(
    model,
    device,
    num_samples=16,
    timesteps=1000,
    img_shape=(3, 32, 32),
):
    """
    DDPM sampling.
    Starts from x_T ~ N(0,I) and applies the reverse process using the
    trained noise predictor epsilon_theta(x_t, t).

    Returns:
        Tensor in [0,1], shape (num_samples, C, H, W).
    """
    model.eval()
    betas, alphas, alpha_bar = _precompute_schedule(T=timesteps, device=device)

    x = torch.randn((num_samples, *img_shape), device=device)  # x_T ~ N(0, I)

    for t in reversed(range(timesteps)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_batch)

        a_t = alphas[t]
        ab_t = alpha_bar[t]
        b_t = betas[t]

        # DDPM mean
        mean = (x - ((1 - a_t) / torch.sqrt(1 - ab_t)) * eps_pred) / torch.sqrt(a_t)

        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(b_t) * noise
        else:
            x = mean

    # Project to valid image range
    x = x.clamp(0.0, 1.0)
    return x