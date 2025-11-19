import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class model(nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()

        # --- DEFINING LAYERS ---
        # Convolutional part
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (N,1,12,12) -> (N,16,12,12)
            nn.LeakyReLU(0.1), #nn.ReLU(), #nn.leakyReL(1.0)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (N,16,12,12) -> (N,32,12,12)
            nn.LeakyReLU(0.1),#nn.ReLU(),
            nn.AvgPool2d(2)                              # (N,32,12,12) -> (N,32,6,6)
        )

        # Fully connected part
        self.fc = nn.Sequential(
            nn.Linear(32 * 6 * 6, 64),
            nn.LeakyReLU(0.1),#nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        
        '''self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 1 -> 8
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 8 -> 16
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(2)                              # 12x12 -> 6x6
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 6 * 6, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.7),
            nn.Linear(16, 1) (32, 1)
        )'''

        '''self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(2)   # -> (N,16,6,6)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 6 * 6, 16),  # 576 -> 16  (~9k param)
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )'''

        # --- WEIGHT LOADING ---
        if load_weights:
            try:
                path_to_py = os.path.dirname(__file__)

                weight_path = os.path.join(path_to_py, "model_weights.pth")

                state = torch.load(weight_path, map_location="cpu")
                self.load_state_dict(state)
                self.eval()

                print(f"Loaded pretrained weights from: {weight_path}")

            except FileNotFoundError:
                print("Warning: model_weights.pth not found, using random weights.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure shape (N,1,12,12)
        if x.ndim == 2:
            # single sample (12,12)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            # batch (N,12,12)
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            # assume (N,1,12,12) already
            pass
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")

        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)             # (N,1)
        return x.squeeze(1)        # (N,)

    def pred(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32) 
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))
