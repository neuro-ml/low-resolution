import os
import random

import numpy as np
import torch


def get_lowres_dir_name():
    return os.path.dirname(__file__)


def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pred(x, threshold=0.5):
    return x[0] > threshold


def np_sigmoid(x):
    """Applies sigmoid function to the incoming value(-s)."""
    return 1 / (1 + np.exp(-x))


def volume2diameter(volume):
    return (6 * volume / np.pi) ** (1 / 3)
