import math
import sys

import mila_tools

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys() or __debug__

name = 1
age = 2

SWEEP = False  # "sweep_toy.yaml"
CLUSTER = 1

RANDOM_SEED = 1337

dataset = "iris"
num_hidden = 32
initial_lr_x = .001
initial_lr_y = initial_lr_x * 10.

decay_steps = 500000
decay_factor = .5
blocks = [2, ] * 5


# blocks = [2, 8]

def state_fn(x):
    x = x + x
    return x


use_adam = False
adam1 = 0.9
adam2 = 0.99

batch_size = 32
weight_norm = 0.00

num_epochs = 1000000
eval_every = math.ceil(num_epochs / 1000)

mila_tools.register(locals())

################################################################
# Derivative parameters
################################################################
lr_x = initial_lr_x + 1
lr_y = initial_lr_y + 1

mila_tools.deploy(CLUSTER, SWEEP)
