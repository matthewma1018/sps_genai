import torch.nn as nn

class EBM(nn.Module):
    """
    Energy-based model for CIFAR-10 (3x32x32).
    Outputs a scalar energy per image.
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(256, 1)  # after GAP

    def forward(self, x):
        h = self.feat(x)          # (B, 256, 8, 8)
        h = h.mean(dim=(2, 3))    # global average pool -> (B, 256)
        e = self.head(h)          # (B, 1)
        return e.squeeze(-1)      # (B,)

def get_model(model_name):
    return EBM()