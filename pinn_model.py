import torch
import torch.nn as nn

class PINN(nn.Module):
   
    def __init__(self):
        super(PINN, self).__init__()

       
        self.net = nn.Sequential(
            nn.Linear(4, 256),      # Input layer: 4 features (S, t, sigma, r) -> 64 features
            nn.SiLU(),             # Activation function
            nn.Linear(256, 256),     # Hidden layer 1
            nn.SiLU(),             # Activation function
            nn.Linear(256, 256),     # Hidden layer 2
            nn.SiLU(),             # Activation function
            nn.Linear(256, 256),     # Hidden layer 3
            nn.SiLU(),             # Activation function
            nn.Linear(256, 256),     # Hidden layer 3
            nn.SiLU(),             # Activation function
            nn.Linear(256, 256),     # Hidden layer 3
            nn.SiLU(),             # Activation function
            nn.Linear(256, 1)       # Output layer: 256 features -> 1 feature (V)
        )
    

    def forward(self, x):
        return self.net(x)