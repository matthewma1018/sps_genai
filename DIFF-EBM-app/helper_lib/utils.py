from pathlib import Path
import torch

EBM_CKPT = "checkpoints/ebm.pt"
DIFFUSION_CKPT = "checkpoints/diffusion.pt"

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path=None, model_type=None):
    if path is None:
        if model_type is None:
            raise ValueError("Provide path or model_type ('ebm' or 'diffusion').")
        path = EBM_CKPT if model_type.lower() == "ebm" else DIFFUSION_CKPT
    p = Path(path)
    torch.save(model.state_dict(), str(p))
    print(f"Saved -> {p}")

def load_model(model, path=None, model_type=None, device=None, strict=True, eval_mode=False):
    if path is None:
        if model_type is None:
            raise ValueError("Provide path or model_type ('ebm' or 'diffusion').")
        path = EBM_CKPT if model_type.lower() == "ebm" else DIFFUSION_CKPT
    if device is None:
        device = get_device()
    p = Path(path)
    state = torch.load(str(p), map_location=device)
    model.load_state_dict(state, strict=strict)
    model.to(device)
    if eval_mode:
        model.eval()
    print(f"Loaded <- {p} on {device}")
    return model