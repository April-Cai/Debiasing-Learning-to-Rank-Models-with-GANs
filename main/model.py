import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import tensor
import torch.nn.functional as F
import numpy as np


# generator
class Gen(nn.Module):
    def __init__(self, n_features, layers=[512, 256, 128], temperature=0.2):
        super(Gen, self).__init__()
        layers = layers + [1]
        current_size = n_features
        self.temperature = temperature

        self.fc = nn.Sequential()
        for i in range(len(layers)):
            self.fc.add_module("linear_{}".format(i), nn.Linear(current_size, layers[i]))
            if i != len(layers) - 1:
                self.fc.add_module("relu_{}".format(i), nn.LeakyReLU())
            current_size = layers[i]
        self.fc.add_module("tanh", nn.Tanh())

    def forward(self, input_feature):
        out = self.fc(input_feature).view(len(input_feature), -1)
        return out

    # Eq.(7) d_reward is generated from discriminator
    def score(self, input_feature, d_reward):
        score = self.forward(input_feature) / self.temperature
        softmax_score = torch.nn.functional.softmax(score, dim=-1)
        return -torch.mean(torch.log(softmax_score) * d_reward)


# discriminator
class Dis(nn.Module):
    def __init__(self, n_features, layers=[512, 256, 128]):
        super(Dis, self).__init__()
        self.sig = nn.Sigmoid()
        layers = layers + [1]
        current_size = n_features

        self.fc = nn.Sequential()
        for i in range(len(layers)):
            self.fc.add_module("linear_{}".format(i), nn.Linear(current_size, layers[i]))
            self.fc.add_module("relu_{}".format(i), nn.LeakyReLU())
            current_size = layers[i]

    def forward(self, input_feature):
        out = self.fc(input_feature)
        return out

    def reward(self, input_feature):
        return (self.sig(self.forward(input_feature)) - 0.5) * 2


# discriminative loss
# Eq.(3): true_diff can be regarded as pairwise propensity weight(importance)
def pairwise_loss(pred_diffs, true_diff):
    loss = true_diff * torch.log(1 + torch.exp(-pred_diffs))
    return torch.sum(loss) / torch.sum(true_diff)


# position estimation loss
def estimate_loss(pred_diffs, true_diffs):
    loss = np.log(1 + np.exp(-pred_diffs * true_diffs))
    return loss
