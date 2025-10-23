import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path="model.pth"):
    print(f"Saving model to {path}...")
    torch.save(model.state_dict(), path)

def load_model(model, path="model.pth"):
    print(f"Loading model from {path}...")
    model.load_state_dict(torch.load(path))
    return model