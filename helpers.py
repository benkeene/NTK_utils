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


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_list=None):
        [data_X, data_y] = dataset
        X_tensor, y_tensor = data_X.clone().detach(), data_y.clone().detach()
        tensors = (X_tensor, y_tensor)
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transforms:
            x = self.transforms(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
