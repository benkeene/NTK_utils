import matplotlib.pyplot as plt
import network
import torch
import torch.nn as nn
import torch.optim as optim
from functorch import make_functional
import math
import IPython
from helpers import empirical_ntk_jacobian_contraction, get_weights_copy, CustomDataset

DEBUG = 0


def fnet_single(params, x):  # evaluates the model at a single data point
    return fnet(params, x.unsqueeze(0)).squeeze(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

n_points = 20

gamma_list = torch.linspace(-3.14, 3.14, n_points)
x0_list = torch.as_tensor([torch.cos(gamma) for gamma in gamma_list])
x1_list = torch.as_tensor([torch.sin(gamma) for gamma in gamma_list])
x = torch.stack([x0_list, x1_list], dim=1)
f_x0x1 = x0_list * x1_list


if DEBUG:
    assert len(x0_list) == len(x1_list) == len(
        f_x0x1), 'dimension mismatch in dataset construction'

dataset = CustomDataset([x, f_x0x1])


train_loader = torch.utils.data.DataLoader(
    dataset)

# SPECIFIED BY PAPER
depth = 4
loss_fn = nn.MSELoss()


size = len(train_loader.dataset)  # number of data points

colors = [
    '#cc241d',
    '#98971a',
    '#d79921',
    '#458588',
    '#b16286',
    '#689d6a'
]

plt.style.use('gruvbox.mplstyle')

n_inits = 3

width_list = [50, 500, 1000, 1500, 2000, 2500]

gamma_fixed = torch.tensor(3.14 / 4)

x21 = torch.cos(gamma_fixed)
x22 = torch.sin(gamma_fixed)

x2_fixed = torch.as_tensor([x21.clone().detach(), x22.clone().detach()])

for j, width in enumerate(width_list):
    print(f'width: {width}')
    for i in range(n_inits):
        print(f'init: {i+1}/{n_inits}')
        # specified by paper
        layer_widths = [2, *[width for _ in range(depth - 2)], 1]
        model = network.LinearNet(layer_widths).to(
            device)

        fnet, params = make_functional(model)

        optimizer = optim.SGD(model.parameters(), lr=1)
        model.train()

        for batch, (X, y) in enumerate(train_loader):

            X, y = X.to(device), y.to(device)
            if batch == 0:

                model_copy = network.LinearNet(layer_widths).to(device)
                model_copy.load_state_dict(get_weights_copy(model))
                fnet, params = make_functional(model_copy)

                x1 = X.clone().detach()
                x2 = X.clone().detach()

                x2[0] = x2_fixed.clone().detach()

                NTKlist = []
                for gamma in gamma_list:

                    x1[0] = torch.as_tensor(
                        [torch.cos(gamma), torch.sin(gamma)])

                    NTK = empirical_ntk_jacobian_contraction(
                        fnet_single, params, x1, x2)

                    NTKlist.append(NTK[0][0][0].item())

                plt.plot(gamma_list, NTKlist, ':',
                         label=f'w = {width}, n = {batch}' if i == 0 else None, alpha=0.5, color=colors[j])

                if i == 0:
                    plt.legend()

            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if batch == n_points - 1:  # last batch, same code as NTK on first batch

                model_copy = network.LinearNet(layer_widths).to(device)
                model_copy.load_state_dict(get_weights_copy(model))
                fnet, params = make_functional(model_copy)

                x1 = X.clone().detach()
                x2 = X.clone().detach()

                x2[0] = x2_fixed.clone().detach()

                NTKlist = []
                for gamma in gamma_list:

                    x1[0] = torch.as_tensor(
                        [torch.cos(gamma), torch.sin(gamma)])

                    NTK = empirical_ntk_jacobian_contraction(
                        fnet_single, params, x1, x2)

                    NTKlist.append(NTK[0][0][0].item())
                plt.plot(gamma_list, NTKlist, '-',
                         label=f'w = {width}, n = {batch+1}' if i == 0 else None, alpha=0.5, color=colors[j])

                if i == 0:
                    plt.legend()

plt.title('NTK^(4) (x_0, x) vs gamma' +
          '\n x_0 = (1/sqrt(2), 1/sqrt(2)), x = (cos(gamma), sin(gamma))')
plt.xlabel('gamma')
plt.ylabel('NTK^(4) (x_0, x)')
plt.show()
