from torch import nn


class VelocityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VelocityPredictor, self).__init__()
        self.pointwise_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x):
        B, N, D = x.shape
        x = self.pointwise_mlp(x)  # (B, N, D) -> (B, N, hidden_size)
        x = x.max(dim=1)  # (B, N, hidden_size) -> (B, hidden_size)
        x = self.global_mlp(x[0])  # (B, hidden_size) -> (B, 2)
        return x
