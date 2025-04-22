import torch.nn as nn

class MlpMinigridPolicy(nn.Module):
    def __init__(self, num_actions=7):
        super().__init__()
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*7**2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(dim=0)
            
        features = self.features(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values