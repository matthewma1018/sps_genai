import torch

@torch.no_grad()
def _init_noise(num_samples, device, shape=(3, 32, 32)):
    # Start from uniform noise in [0,1]
    return torch.rand((num_samples, *shape), device=device)

def generate_samples(
    model,
    device,
    num_samples=16,
    steps=100,
    step_size=0.1,
    noise_scale=0.01,
    clamp=True,
):
    """
    Langevin sampler for an Energy-Based Model E(x).
    Update: x <- x - (step_size^2 / 2) * ∇E(x) + step_size * N(0, I)

    Args:
        model: EBM, maps (B,3,32,32) -> (B,) energy.
        device: "cpu" or "cuda".
        num_samples: how many images to sample.
        steps: number of Langevin steps.
        step_size: step size (epsilon).
        noise_scale: stochastic noise multiplier (default ties to step_size).
        clamp: clamp to [0,1] after each step.

    Returns:
        Tensor of sampled images in [0,1], shape (num_samples, 3, 32, 32).
    """
    model.eval()

    # Disable param grads to save memory/compute
    requires = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    x = _init_noise(num_samples, device)  # [0,1]
    s = torch.tensor(step_size, device=device)
    sigma = torch.tensor(noise_scale, device=device)

    for _ in range(steps):
        x.requires_grad_(True)
        energy = model(x).sum()                 # scalar
        grad_x = torch.autograd.grad(energy, x, create_graph=False)[0]
        with torch.no_grad():
            x = x - 0.5 * (s * s) * grad_x + s * sigma * torch.randn_like(x)
            if clamp:
                x = x.clamp(0.0, 1.0)

    # Restore param flags
    for p, r in zip(model.parameters(), requires):
        p.requires_grad_(r)

    return x.detach()