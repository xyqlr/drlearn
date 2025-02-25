import torch
import torch.nn as nn
import torch.nn.functional as F
from rlearn.nnet import NeuralNetModel

class BlackJackModel(NeuralNetModel):
    def __init__(self, game, args):
        self.state_size = game.get_shape()[0] + 1
        self.action_size = game.get_action_size()
        super().__init__(game, args, (self.state_size,))
        self.fc1 = nn.Linear(self.state_size, args.num_channels)
        self.fc2 = nn.Linear(args.num_channels, args.num_channels)
        self.fc3 = nn.Linear(args.num_channels, self.action_size)
        self.fc4 = nn.Linear(args.num_channels, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        pi = self.fc3(x)
        v = self.fc4(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)