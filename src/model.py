import torch
import numpy as np
import sklearn
import torch.nn as nn
import torch.nn.functional as F

'''
Note: A lot from chat, just to get the structure. Will have to adjust. Dont think weights
is handled correctly
'''

class model:
    def __init__(self):
        M = 12
        input_dim = M*M #(for linear layer)
        hidden_dim = 128
        
        # Neural network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # output in [0,1] #want 5 layers approx
        )
        
        # Load pre-trained weights if you’ve trained the model already
        try:
            self.net.load_state_dict(torch.load("models/model_weights.pth", map_location=torch.device('cpu')))
            self.net.eval()
        except FileNotFoundError:
            print("Warning: model_weights.pth not found, using random weights.")
    
    def pred(self, x):
        if x.ndim > 2: #flatten input
            x = x.view(x.size(0), -1)
        
        with torch.no_grad():
            y_pred = self.net(x).squeeze()  # shape (N,)
        return y_pred
    
    '''
    class IsingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # output: 16x12x12
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # output: 32x12x12
        self.pool = nn.AvgPool2d(2)                               # output: 32x6x6
        self.fc1 = nn.Linear(32*6*6, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # shape: (N, 1, 12, 12)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)).squeeze(1)
        return x

    '''