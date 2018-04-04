import torch
import data

name = "mlp"


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.BatchNorm1d(data.TIMESTEPS * data.FEATURES),
            torch.nn.Linear(data.TIMESTEPS * data.FEATURES, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, data.FEATURES),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, data.TIMESTEPS * data.FEATURES)
        return self.seq(x)