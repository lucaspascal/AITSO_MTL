import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_blocks(model, inst_type):
    """
    Returns all instance of the requested type in a model.
    """
    if isinstance(model, inst_type):
        blocks = [model]
    else:
        blocks = []
        for child in model.children():
            blocks += get_blocks(child, inst_type)
    return blocks



def create_optimizer(opt_type, params, lr):
    """
    Creates the desired optimizer.
    """
    if opt_type.upper()=='ADAM':
        opt = optim.Adam(params, lr=lr)
    elif opt_type.upper()=='SGD':
        opt = optim.SGD(params, lr=lr)
    return opt


