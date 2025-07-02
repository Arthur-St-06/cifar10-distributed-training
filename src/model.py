import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # output: 32×32×32
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → 32×16×16

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → 64×16×16
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → 64×8×8

            nn.Flatten(),                                # → 64*8*8 = 4096
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)
