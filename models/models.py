import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_hidden_units_1, num_hidden_units_2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, num_hidden_units_1),
            nn.ReLU(),
            nn.Linear(num_hidden_units_1, num_hidden_units_2),
            nn.ReLU(),
            nn.Linear(num_hidden_units_2, 1),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_, output):
        predicted_output = self.mlp(input_[None])[0, 0]
        return (output - predicted_output) ** 2
