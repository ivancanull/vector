from matplotlib import axes
import test
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from src import *

def read_touchstone():
    touchstone = TouchstoneParser(2)
    data = touchstone.parse("data/coupled_line_filter_touchstone.s2p")
    Z = FunctionFromTouchstone(2, [0, 0])
    Z.parse("data/coupled_line_filter_touchstone.s2p")
    return Z

def train_bnn(net: nn.Module, x: torch.Tensor, y: torch.Tensor):

    if torch.cuda.is_available():
        net = net.cuda()
        x = x.cuda()
        y = y.cuda()

    mse_loss = nn.MSELoss()
    kl_loss = BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.0001

    optimizer = optim.Adam(net.parameters(), lr=.01)
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

    return net

def test_bnn(net: nn.Module, x: torch.Tensor, samples: int):

    n_freq = len(x)
    y_samp = np.zeros((samples, n_freq))
    if torch.cuda.is_available():
        x = x.cuda()

    for s in range(samples):
        y_tmp = net(x).cpu().detach().numpy()
        y_samp[s] = y_tmp.reshape(-1)
    
    return y_samp
    
def bnn_sp_demo_s11_abs():
    
    ###########################################
    #  1. just train the entire |S11| values  #
    ###########################################
    Z = read_touchstone()
    fmin = 3.0e9 
    fmax = 4.2e9
    freq = np.arange(3, 4.21, 0.01) * 1e9
    
    # sample a few 
    s = frequency_to_s(freq)
    values = Z(s)
    abs_values = np.abs(values) * 10

    norm_freq = freq / 1e9
    # train the bnn
    net = nn.Sequential(
        BayesianLinear(prior_mu=0, prior_sigma=.1, in_features=1, out_features=100),
        nn.Sigmoid(),
        BayesianLinear(prior_mu=0, prior_sigma=.1, in_features=100, out_features=100),
        nn.Sigmoid(),
        BayesianLinear(prior_mu=0, prior_sigma=.1, in_features=100, out_features=1)
    )

    norm_freq = 2 * (freq - freq.min()) / (freq.max() - freq.min()) - 1

    x = torch.tensor(norm_freq, dtype=torch.float).reshape(-1, 1)
    y = torch.tensor(abs_values, dtype=torch.float).reshape(-1, 1)
    net = train_bnn(net, x, y)

    # test on different samples
    x = torch.tensor(norm_freq, dtype=torch.float).reshape(-1, 1)
    y_samp = test_bnn(net, x, samples=100)

    # plot the true value and samples
    import matplotlib.pyplot as plt
    plt.scatter(freq, abs_values, color='red', label='True')
    plt.plot(freq, y_samp.T, alpha=0.1)
    plt.legend()
    plt.savefig("bnn_sp_demo.png")

def bnn_sp_demo_s11_ri():

    #########################################
    #  2. train S11_R and S11_I seperately  #
    #########################################

    Z = read_touchstone()
    fmin = 3.0e9 
    fmax = 4.2e9
    freq = np.arange(3, 4.21, 0.01) * 1e9
    
    # sample a few 
    train_freq = freq[::5] # use every 10th point
    
    real_values_train = np.real(Z(frequency_to_s(train_freq)))
    imag_values_train = np.imag(Z(frequency_to_s(train_freq)))

    real_values_test = np.real(Z(frequency_to_s(freq)))
    imag_values_test = np.imag(Z(frequency_to_s(freq)))

    norm_freq = 2 * (freq - freq.min()) / (freq.max() - freq.min()) - 1
    norm_train_freq = 2 * (train_freq - freq.min()) / (freq.max() - freq.min()) - 1
    ### real part
    # train the bnn
    real_net = nn.Sequential(
        BayesianLinear(prior_mu=0, prior_sigma=.01, in_features=1, out_features=256),
        nn.Softplus(),
        BayesianLinear(prior_mu=0, prior_sigma=.01, in_features=256, out_features=256),
        nn.Sigmoid(),
        BayesianLinear(prior_mu=0, prior_sigma=.01, in_features=256, out_features=1)
    )

    

    x = torch.tensor(norm_train_freq, dtype=torch.float).reshape(-1, 1)
    y = torch.tensor(real_values_train, dtype=torch.float).reshape(-1, 1)
    real_net = train_bnn(real_net, x, y)
    
    ### imag part
    # train the bnn
    imag_net = nn.Sequential(
        BayesianLinear(prior_mu=0, prior_sigma=.01, in_features=1, out_features=256),
        nn.Softplus(),
        BayesianLinear(prior_mu=0, prior_sigma=.01, in_features=256, out_features=256),
        nn.Sigmoid(),
        BayesianLinear(prior_mu=0, prior_sigma=.01, in_features=256, out_features=1)
    )

    x = torch.tensor(norm_train_freq, dtype=torch.float).reshape(-1, 1)
    y = torch.tensor(imag_values_train, dtype=torch.float).reshape(-1, 1)
    imag_net = train_bnn(imag_net, x, y)

    # test on different samples
    x = torch.tensor(norm_freq, dtype=torch.float).reshape(-1, 1)
    y_samp_r = test_bnn(real_net, x, samples=100)
    y_samp_i = test_bnn(imag_net, x, samples=100)
    y_samp_abs = np.abs(y_samp_r + 1j * y_samp_i)
    y_samp_dB = 20 * np.log10(y_samp_abs)
    y_samp_dB_var = np.var(y_samp_dB, axis=0)
    # plot the true value and samples
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))

    # plot the real part
    plt.subplot(411)
    plt.scatter(freq, real_values_test, color='red', label='True Real Part')
    plt.plot(freq, y_samp_r.T, alpha=0.1)
    plt.legend()

    # plot the imaginary part
    plt.subplot(412)
    plt.scatter(freq, imag_values_test, color='blue', label='True Imag Part')
    plt.plot(freq, y_samp_i.T, alpha=0.1)
    plt.legend()

    # plot the absolute value using dB
    plt.subplot(413)
    plt.scatter(freq, 20 * np.log10(np.abs(real_values_test + 1j * imag_values_test)), color='green', label='True Abs Value (dB)')
    plt.plot(freq, y_samp_dB.T, alpha=0.1)
    plt.plot(freq, y_samp_dB.T, alpha=0.1)

    # plot the variance of the samples
    plt.subplot(414)
    plt.plot(freq, y_samp_dB_var, color='orange', label='Variance of Samples')

    plt.legend()
    # save the figure
    plt.savefig("bnn_sp_demo_real_imag_abs_Softplus_Sigmoid.png")
    

if __name__ == "__main__":
    # bnn_sp_demo_s11_abs()
    bnn_sp_demo_s11_ri()