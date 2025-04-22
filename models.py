import torch.nn as nn

class MlpMinigridPolicy(nn.Module):
    def __init__(self, num_actions=7):
        super().__init__()
        self.num_actions = num_actions
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*7**2, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(dim=0)
        return self.fc(x)