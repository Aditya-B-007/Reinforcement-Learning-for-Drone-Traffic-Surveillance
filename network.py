import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, atoms, args):
        super(DQN, self).__init__()
        self.atoms = atoms
        self.action_dim = action_dim
        self.use_dueling = args.use_dueling_network

        # Common feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        if self.use_dueling:
            # Dueling architecture: value and advantage streams
            self.value_stream = nn.Linear(128, atoms)
            self.advantage_stream = nn.Linear(128, action_dim * atoms)
        else:
            # Standard architecture
            self.fc_out = nn.Linear(128, action_dim * atoms)

    def forward(self, x):
        features = self.feature_layer(x)

        if self.use_dueling:
            value = self.value_stream(features).view(-1, 1, self.atoms)
            advantage = self.advantage_stream(features).view(-1, self.action_dim, self.atoms)
            # Combine value and advantage: Q = V + (A - mean(A))
            q_dist = value + advantage - advantage.mean(1, keepdim=True)
        else:
            q_dist = self.fc_out(features).view(-1, self.action_dim, self.atoms)

        # Apply softmax to get the probability distribution for each action's value
        return F.softmax(q_dist, dim=2)