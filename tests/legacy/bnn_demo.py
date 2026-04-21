from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src import *

test_BayesianLinear() 

def toy_function(x):
    return -x ** 4 + 3 * x ** 2 + 1

# toy dataset we can start with
x = [-2, -1.8, -1, 1, 1.8, 2]
# x = np.linspace(-2, 2, 10)
x = torch.tensor(x, dtype=torch.float).reshape(-1,1)
y = toy_function(x)

net = nn.Sequential(
    BayesianLinear(prior_mu=0, prior_sigma=1, in_features=1, out_features=32),
    nn.Sigmoid(),
    BayesianLinear(prior_mu=0, prior_sigma=1, in_features=32, out_features=1)
)

mse_loss = nn.MSELoss()
kl_loss = BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01


if torch.cuda.is_available():
    net = net.cuda()
    x = x.cuda()
    y = y.cuda()


optimizer = optim.Adam(net.parameters(), lr=0.1)
epochs = 2000
# wrap with tqdm
with tqdm(total=epochs, unit="epoch") as pbar:
    for i in range(epochs):
        pred = net(x)
        loss = mse_loss(pred, y) + kl_weight * kl_loss(net)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the progress bar
        if i % 10 == 9:
            pbar.set_postfix_str(f'Epoch {i+1}/{epochs}, Loss: {loss.item():.4f}')
            pbar.update(10)

samples = 100
x_tmp = torch.linspace(-3,3,100).reshape(-1,1)
y_tmp_true = toy_function(x_tmp)
y_samp = np.zeros((samples,100))

if torch.cuda.is_available():
    x_tmp = x_tmp.cuda()

for s in range(samples):
    y_tmp = net(x_tmp).cpu().detach().numpy()
    y_samp[s] = y_tmp.reshape(-1)

# plot x_tmp and y_samp
import matplotlib.pyplot as plt
plt.scatter(x.cpu().detach().numpy(), y.cpu().detach().numpy(), color='red', label='data')
plt.plot(x_tmp.cpu().detach().numpy(), y_tmp_true, label='true function')
plt.plot(x_tmp.cpu().detach().numpy(), y_samp.T, alpha=0.1)
plt.legend()
plt.savefig("bnn_demo.png")
print("test result:",np.mean(y_samp, axis = 0))