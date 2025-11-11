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
        # Define network architecture directly here
        # Adjust dimensions to match your training data
        input_dim = 400  # for example, if M=20 → 20×20 = 400
        hidden_dim = 128
        
        # Build a small neural network
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
        # Flatten input if necessary (e.g., from (N, M, M) → (N, M*M))
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        
        with torch.no_grad():
            y_pred = self.net(x).squeeze()  # shape (N,)
        return y_pred