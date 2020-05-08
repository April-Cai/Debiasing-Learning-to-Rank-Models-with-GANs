import numpy as np
import random

# data
data = 'S'
y_dir = '../data/Yahoo'
s_dir = '../data/sample'
tools_dir = '../tools/trec_eval'

n_positions = 10
n_features = 700
batch_size = 5
n_cpu = 4

# training
seed = 2019
n_epochs = 2


d_lr = 0.02
g_lr = d_lr
rel_hidden = 256
prop_hidden = 4

d_epochs = 20
g_epochs = 5
