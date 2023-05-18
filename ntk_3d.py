import matplotlib.pyplot as plt
import network

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functorch import make_functional, make_functional_with_buffers, vmap, vjp, jvp, jacrev
import IPython
import copy
import math


def fnet_single(params, x):  # evaluates the model at a single data point
    return fnet(params, x.unsqueeze(0)).squeeze(0)


# computes the NTK between two data points
# for our purposes, x2 is fixed and x1 is the input
def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2)
                         for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result


def get_weights_copy(model):
    weights_path = 'weights.pth'
    torch.save(model.state_dict(), weights_path)
    return torch.load(weights_path)
