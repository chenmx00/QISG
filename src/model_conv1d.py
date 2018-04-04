import torch
import data

name = "conv1d"
LINEAR_IN = (data.TIMESTEPS - 8) * data.FEATURES


class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.seq1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=data.FEATURES, out_channels=32, kernel_size=5),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=32, out_channels=data.FEATURES, kernel_size=5),
            torch.nn.BatchNorm1d(data.FEATURES),
            torch.nn.ReLU())
        self.seq2 = torch.nn.Sequential(
            torch.nn.Linear(LINEAR_IN,
                            data.FEATURES), torch.nn.Tanh())

    def forward(self, x):
        x = self.seq1(x)
        x = x.view(-1, LINEAR_IN)
        return self.seq2(x)