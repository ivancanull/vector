# a demo to predict S11_R and S11_I using a Bayesian Neural Network

# 1. the model takes input features including geo params and frequency
from src import *
from torch.utils.data import random_split
from torch import nn, optim
import matplotlib.pyplot as plt

RETRAIN = False

def min_max_normalizer(x, min_x, max_x):
    """
    Normalize the input features and target values using min-max normalization.
    
    Args:
    - x (torch.Tensor): Input features.
    - min_x (list): Minimum values for normalization.
    - max_x (list): Maximum values for normalization.
    
    Returns:
    - x_norm (torch.Tensor): Normalized input features.
    - y_norm (torch.Tensor): Normalized target values.
    """
    min_x = torch.tensor(min_x, dtype=torch.float32, device=x.device)
    max_x = torch.tensor(max_x, dtype=torch.float32, device=x.device)
    x_norm = (x - min_x) / (max_x - min_x)
    return x_norm

def train_bnn(net: nn.Module, x: torch.Tensor, y: torch.Tensor, normalize):

    if torch.cuda.is_available():
        net = net.cuda()
        x = x.cuda()
        y = y.cuda()

    if normalize:
        # define the min and max values for normalization
        MIN_N = [50, 1.0, 10, 0.1e9]
        MAX_N = [70, 2.0, 20, 19.99e9]
        x = min_max_normalizer(x, MIN_N, MAX_N)

    mse_loss = nn.MSELoss()
    kl_loss = BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.0001

    optimizer = optim.Adam(net.parameters(), lr=.01)
    epochs = 4000
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

def bnn_geo_params():

    # define the geometric parameters and frequency points min and max values for normalization
    MIN_N = [50, 1.0, 10, 0.1e9]
    MAX_N = [70, 2.0, 20, 19.99e9]
    freqs = np.arange(0.1e9, 20e9, 0.03e9)

    # create dataset 
    dataset_path = "/public/home/zhanghf/projects/vector/data/spiral_inductor_s_parameters_export"
    port_num = 2
    dataset = SPDataset(dataset_path, freqs, port_num, [0, 0], "abs")
    
    # split train / test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    test_dataset.random_sample = False  # disable random sampling for test dataset
    
    # create dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # create model
    net = nn.Sequential(
        BayesianLinear(prior_mu=0, prior_sigma=.1, in_features=4, out_features=256),
        nn.Softplus(),
        BayesianLinear(prior_mu=0, prior_sigma=.1, in_features=256, out_features=256),
        nn.Sigmoid(),
        BayesianLinear(prior_mu=0, prior_sigma=.1, in_features=256, out_features=1)  # S11_R and S11_I
    )
    if torch.cuda.is_available():
        net = net.cuda()
    
    if os.path.exists('bnn_geo_params.pth') and not RETRAIN:
        print("Loading pre-trained model...")
        net.load_state_dict(torch.load('bnn_geo_params.pth'))
    else:
        print("Training new model...")
        # train the model
        epochs = 4000
        mse_loss = nn.MSELoss()
        kl_loss = BKLLoss(reduction='mean', last_layer_only=False)
        kl_weight = 0.1
        optimizer = optim.Adam(net.parameters(), lr=.001)

        with tqdm(total=epochs, unit="epoch") as pbar:
            for epoch in range(epochs):
                for x, y in train_loader:
                    x = min_max_normalizer(x, MIN_N, MAX_N)
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()
                    # forward pass
                    pred = net(x)
                    # compute loss
                    loss = mse_loss(pred, y) + kl_weight * kl_loss(net)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if epoch % 10 == 9:
                    # update the progress bar
                    pbar.set_postfix_str(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
                    pbar.update(10)
        # save the model
        torch.save(net.state_dict(), 'bnn_geo_params.pth')

    # test the model
    samples = 100
    net.eval()
    with torch.no_grad():
        idx = 0
        for x, y in test_loader:
            x = min_max_normalizer(x, MIN_N, MAX_N)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()    
            y_samp = np.zeros((samples, len(freqs)))
            for s in range(samples):
                y_tmp = net(x).cpu().detach().numpy()
                y_samp[s] = y_tmp.reshape(-1)
            # plot the results
            y_samp = y_samp.T
            y_samp_var = np.var(y_samp, axis=1)
            plt.figure(figsize=(10, 5))
            plt.scatter(freqs, y.cpu().numpy(), label='True S11 Abs', color='blue')
            plt.plot(freqs, y_samp, alpha=0.1)
            plt.fill_between(freqs, y_samp.min(axis=1), y_samp.max(axis=1), alpha=0.2, color='red', label='Range')
            plt.savefig(f'final/bnn_geo_params_s11_abs_{idx}.png')
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(freqs, y_samp_var, label='Aleatoric Uncertainty', color='red')
            plt.savefig(f'final/bnn_geo_params_s11_abs_noise_{idx}.png')
            plt.close()

            idx += 1


    

if __name__ == "__main__":
    bnn_geo_params()