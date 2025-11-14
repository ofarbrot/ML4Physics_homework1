import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class model(nn.Module):
    def __init__(self):
        super().__init__()

        M = 12  # dont know if I need

        # Convolutional part
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (N,1,12,12) -> (N,16,12,12)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (N,16,12,12) -> (N,32,12,12)
            nn.ReLU(),
            nn.AvgPool2d(2)                              # (N,32,12,12) -> (N,32,6,6)
        )

        # Fully connected part
        self.fc = nn.Sequential(
            nn.Linear(32 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #nn.Sigmoid()  # output in [0,1]
        )
        #Try to make it work saving the weights: 
        # Try loading pretrained weights (optional)
        try:
            # Build an absolute path to the weights file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(current_dir, "...", "modells", "model_weights.pth")

            # Normalize the path (makes it OS-safe)
            weights_path = os.path.normpath(weights_path)

            # Load and apply the weights
            state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
            self.load_state_dict(state_dict)
            self.eval()

            print(f"Loaded pretrained weights from: {weights_path}")

        except FileNotFoundError:
            print(" Warning: model_weights.pth not found, using random weights.")

        """# Try loading pretrained weights (optional)
        try:
            state_dict = torch.load("models/model_weights.pth",
                                    map_location=torch.device("cpu"))
            self.load_state_dict(state_dict)
            self.eval()
        except FileNotFoundError:
            print("Warning: model_weights.pth not found, using random weights.")
"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward: used if you train the model yourself.
        """
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
        """
        Required by the platform: takes input like training data and
        returns 1D tensor of predictions.
        """
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# hi