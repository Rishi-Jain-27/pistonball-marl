
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_conv_dim=32, hidden_head_dim=256):
        super().__init__()

        # Use a conv net because pistonball is just image data sadly
        h, w, c = state_dim

        self.conv_net = nn.Sequential(
            nn.Conv2d(c, hidden_conv_dim, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(hidden_conv_dim, hidden_conv_dim * 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_conv_dim * 2, hidden_conv_dim * 2, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Find output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_size = self.conv_net(dummy).flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, hidden_head_dim),
            nn.ReLU(),
            nn.Linear(hidden_head_dim, action_dim)
        )

    def forward(self, x):
        return self.head(self.conv_net(x.permute(0, 3, 1, 2).float() / 255.0))