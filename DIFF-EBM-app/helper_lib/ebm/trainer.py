import torch
import torch.nn as nn

@torch.no_grad()
def _sample_negatives_like(x):
    neg = torch.rand_like(x)
    return neg

def train_ebm(model, data_loader, optimizer, device="cuda", epochs=10, neg_ratio=1):
    """
    EBM training with Noise-Contrastive Estimation (NCE).
    - Positives: real CIFAR-10 images from the loader.
    - Negatives: uniform-noise images.
    - Energy E(x) -> use logits = -E(x); BCE targets: real=1, fake=0.

    Args:
        model: EBM that maps (B,3,32,32) -> (B,) energies.
        data_loader: yields (images, labels); labels are ignored.
        optimizer: torch optimizer on model parameters.
        device: "cpu" or "cuda".
        epochs: training epochs.
        neg_ratio: number of negatives per real (int).

    Returns:
        model (trained in-place).
    """
    model.to(device).train()
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for imgs, _ in data_loader:
            imgs = imgs.to(device)  # [0,1], shape (B,3,32,32)
            B = imgs.size(0)

            # negatives
            negs = _sample_negatives_like(imgs).to(device)
            if neg_ratio > 1:
                negs = negs.repeat(neg_ratio, 1, 1, 1)  # more negatives than reals

            # energies -> logits = -E(x)
            e_pos = model(imgs)                 # (B,)
            e_neg = model(negs)                 # (B*neg_ratio,)

            logits = torch.cat([-e_pos, -e_neg], dim=0)
            targets = torch.cat([
                torch.ones_like(e_pos),
                torch.zeros_like(e_neg)
            ], dim=0)

            loss = bce(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # tiny print
        print(f"epoch {epoch+1}/{epochs} - nce_loss: {loss.item():.4f}")

    return model