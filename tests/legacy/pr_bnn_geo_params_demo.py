from src import *
from torch.utils.data import random_split
from torch import nn, optim
import matplotlib.pyplot as plt

RETRAIN = True

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

def divide_normalizer(y, normalizer):
    """
    Normalize the target values using the provided normalizer.
    
    Args:
    - y (torch.Tensor): Target values.
    - normalizer (list): Normalization factors for each output feature.
    
    Returns:
    - y_norm (torch.Tensor): Normalized target values.
    """
    normalizer = torch.tensor(normalizer, dtype=torch.float32, device=y.device)
    y_norm = y / normalizer
    return y_norm

def divide_denormalizer(y, normalizer):
    """
    Denormalize the target values using the provided normalizer.
    
    Args:
    - y (torch.Tensor): Normalized target values.
    - normalizer (list): Normalization factors for each output feature.
    
    Returns:
    - y_denorm (torch.Tensor): Denormalized target values.
    """
    normalizer = torch.tensor(normalizer, dtype=torch.float32, device=y.device)
    y_denorm = y * normalizer
    return y_denorm

def evaluate_with_net_output(poles, residues, d, h, freqs, n_poles_real, n_poles_complex):
    """
    Evaluate the S-parameters from poles and residues.
    
    Args:
    - poles (np.ndarray): Poles of the system.
    - residues (np.ndarray): Residues of the system.    
    - d (float): Denominator coefficient.
    - h (float): High-frequency coefficient.
    - freqs (np.ndarray): Frequency points at which to evaluate the S-parameters.
    - n_poles_real (int): Number of real poles.
    - n_poles_complex (int): Number of complex poles pairs.
    
    Returns:
    - s_params (np.ndarray): Evaluated S-parameters.
    """

    s = frequency_to_s(freqs)
    size = poles.shape[0]
    new_poles = np.zeros((size, n_poles_real + 2 * n_poles_complex), dtype=np.complex128)
    new_residues = np.zeros((size, n_poles_real + 2 * n_poles_complex), dtype=np.complex128)
    
    new_poles[:, :n_poles_real] = poles[:, :n_poles_real]
    new_poles[:, n_poles_real:n_poles_real+n_poles_complex*2:2] = poles[:, n_poles_real:n_poles_real + n_poles_complex] + 1j * poles[:, n_poles_real+n_poles_complex:n_poles_real+n_poles_complex*2]
    new_poles[:, n_poles_real+1:n_poles_real+n_poles_complex*2+1:2] = poles[:, n_poles_real:n_poles_real + n_poles_complex] - 1j * poles[:, n_poles_real+n_poles_complex:n_poles_real+n_poles_complex*2]

    new_residues[:, :n_poles_real] = residues[:, :n_poles_real]
    new_residues[:, n_poles_real:n_poles_real+n_poles_complex*2:2] = residues[:, n_poles_real:n_poles_real + n_poles_complex] + 1j * residues[:, n_poles_real+n_poles_complex:n_poles_real+n_poles_complex*2]
    new_residues[:, n_poles_real+1:n_poles_real+n_poles_complex*2+1:2] = residues[:, n_poles_real:n_poles_real + n_poles_complex] - 1j * residues[:, n_poles_real+n_poles_complex:n_poles_real+n_poles_complex*2]

    s_params = []

    d = d.reshape(-1).tolist()
    h = h.reshape(-1).tolist()
    for i in range(size):
        print(new_poles[i], new_residues[i], d[i], h[i])
        s_params.append(evaluate(new_poles[i].tolist(), new_residues[i].tolist(), d[i], h[i], s))

    s_params = np.array(s_params) # shape (size, len(freqs))
    return s_params

MIN_N = [50, 1.0, 10]
MAX_N = [70, 2.0, 20]

NORMALIZER = {}
NORMALIZER["p"] = [1e9, 1e10, 1e10, 1e11, 1e11, 1e11, 1e11, 1e11]
NORMALIZER["r"] = [1e6, 1e8, 1e9, 1e11, 1e11, 1e12, 1e11, 1e11]
NORMALIZER["d"] = [1]
NORMALIZER["h"] = [1e-13]

def train_bnn(train_dataset: PolesResiduesDataset, mode: str, poles_residues_features: int):

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    # create model
    # net = nn.Sequential(
    #     BayesianLinear(prior_mu=0, prior_sigma=.1, in_features=3, out_features=256),
    #     nn.Softplus(),
    #     BayesianLinear(prior_mu=0, prior_sigma=.1, in_features=256, out_features=256),
    #     nn.Sigmoid(),
    #     BayesianLinear(prior_mu=0, prior_sigma=.1, in_features=256, out_features=poles_residues_features)  # S11_R and S11_I
    # )

    # create a mlp model
    net = nn.Sequential(
        nn.Linear(3, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, poles_residues_features)  # S11_R and S11_I
    )

    if torch.cuda.is_available():
        net = net.cuda()

    if os.path.exists(f'pr_bnn_geo_params_{mode}_mlp.pth') and not RETRAIN:
        print("Loading pre-trained model...")
        net.load_state_dict(torch.load(f'pr_bnn_geo_params_{mode}_mlp.pth'))
    else:
        print(f"Training new model with mode {mode}...")
        # train the model
        epochs = 4000
        mse_loss = nn.MSELoss()
        # kl_loss = BKLLoss(reduction='mean', last_layer_only=False)
        # kl_weight = 0.001
        optimizer = optim.Adam(net.parameters(), lr=.001)

        with tqdm(total=epochs, unit="epoch") as pbar:
            for epoch in range(epochs):
                for x, p, r, d, h in train_loader:
                    
                    y = eval(mode)
                    
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()

                    x = min_max_normalizer(x, MIN_N, MAX_N)
                    y = divide_normalizer(y, NORMALIZER[mode])
                    # ensure the input and output shapes match 
                    
                    # forward pass
                    pred = net(x)
                    # compute loss
                    loss = mse_loss(pred, y) # + kl_weight * kl_loss(net)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if epoch % 10 == 9:
                    # update the progress bar
                    pbar.set_postfix_str(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
                    pbar.update(10)
        # save the model
        torch.save(net.state_dict(), f'pr_bnn_geo_params_{mode}_mlp.pth')
    return net

def pr_bnn_geo_params():

    # create dataset
    dataset_path = "/public/home/zhanghf/projects/vector/data/spiral_inductor_s_parameters_export"
    port_num = 2
    freqs = np.arange(0.1e9, 20e9, 0.03e9)
    init_poles = basic_poles_initialize(freqs_range=(0.1e9, 20e9), n_poles_real=4, n_poles_complex=2)
    port_combinations = [(0, 0)]
    n_poles_real = 6
    n_poles_complex = 1
    dataset = PolesResiduesDataset(dataset_path, freqs, port_num, init_poles, n_poles_real, n_poles_complex, port_combinations, random_sample=True)

    # split train / test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)  # for reproducibility
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    test_dataset.random_sample = False  # disable random sampling for test dataset

    poles_residues_features = {
        "p": n_poles_real + 2 * n_poles_complex,  # real_poles, complex_poles are included in the features
        "r": n_poles_real + 2 * n_poles_complex,  # real_residues, complex_residues are included in the features
        "d": 1,
        "h": 1
    }
    nets = {}
    for mode in ["p", "r", "d", "h"]:
        nets[mode] = train_bnn(train_dataset, mode, poles_residues_features[mode])

    samples = 100
    
    # create dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        idx = 0
        for x, p, r, d, h in test_loader:
            y_samp = {}
            x = min_max_normalizer(x, MIN_N, MAX_N)
            y_true = {}
            for mode, net in nets.items():
                if torch.cuda.is_available():
                    x = x.cuda()
                    net = net.cuda()
                # evaluate the model
                net.eval()
                y = net(x)
                y_tmp = divide_denormalizer(y, NORMALIZER[mode]).cpu().detach().numpy()
                y_samp[mode] = y_tmp.reshape(1, -1)
                y_true[mode] = eval(mode).numpy().reshape(1, -1)
            
            s_params = evaluate_with_net_output(
                y_samp["p"], y_samp["r"], y_samp["d"], y_samp["h"],
                freqs, n_poles_real, n_poles_complex
            )
            s_params_true = evaluate_with_net_output(
                y_true["p"], y_true["r"], y_true["d"], y_true["h"],
                freqs, n_poles_real, n_poles_complex
            )

            fig, axes = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)

            axes[0].plot(freqs, np.real(s_params_true.T), label='True S11 real', color='blue')
            axes[0].plot(freqs, np.real(s_params.T), label='Predicted S11 real')
            axes[0].set_xlabel('Frequency (Hz)')
            axes[0].set_ylabel('Real Part')
            axes[0].legend()

            axes[1].plot(freqs, np.imag(s_params_true.T), label='True S11 imag', color='blue')
            axes[1].plot(freqs, np.imag(s_params.T), label='Predicted S11 imag')
            axes[1].set_xlabel('Frequency (Hz)')
            axes[1].set_ylabel('Imaginary Part')
            axes[1].legend()
            
            axes[2].plot(freqs, 20 * np.log10(np.abs(s_params_true.T)), label='True S11 Abs', color='blue')
            axes[2].plot(freqs, 20 * np.log10(np.abs(s_params.T)), label='Predicted S11 Abs')
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_ylabel('Magnitude (dB)')

                
            plt.savefig(f'temp/pr_bnn_geo_params_s11_abs_{idx}_mlp.png')
            plt.close()
            idx += 1
    return

    with torch.no_grad():
        idx = 0
        for x, p, r, d, h in test_loader:
            y_samp = {}
            x = min_max_normalizer(x, MIN_N, MAX_N)
            y_true = {}
            for mode, net in nets.items():
                if torch.cuda.is_available():
                    x = x.cuda()
                    net = net.cuda()
                # evaluate the model
                net.eval()
                y_samp[mode] = np.zeros((samples, poles_residues_features[mode]))
                for s in range(samples):
                    y = net(x)
                    y_tmp = divide_denormalizer(y, NORMALIZER[mode]).cpu().detach().numpy()
                    y_samp[mode][s] = y_tmp.reshape(-1)
                
                y_true[mode] = eval(mode).numpy().reshape(1, -1)
            
            s_params = evaluate_with_net_output(
                y_samp["p"], y_samp["r"], y_samp["d"], y_samp["h"],
                freqs, n_poles_real, n_poles_complex
            )
            s_params_true = evaluate_with_net_output(
                y_true["p"], y_true["r"], y_true["d"], y_true["h"],
                freqs, n_poles_real, n_poles_complex
            )

            fig, axes = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)

            axes[0].scatter(freqs, np.real(s_params_true.T), label='True S11 real', color='blue')
            axes[0].plot(freqs, np.real(s_params.T), alpha=0.1)
            axes[0].set_xlabel('Frequency (Hz)')
            axes[0].set_ylabel('Real Part')
            axes[0].legend()

            axes[1].scatter(freqs, np.imag(s_params_true.T), label='True S11 imag', color='blue')
            axes[1].plot(freqs, np.imag(s_params.T), alpha=0.1)
            axes[1].set_xlabel('Frequency (Hz)')
            axes[1].set_ylabel('Imaginary Part')
            axes[1].legend()
            
            axes[2].scatter(freqs, 20 * np.log10(np.abs(s_params_true.T)), label='True S11 Abs', color='blue')
            axes[2].plot(freqs, 20 * np.log10(np.abs(s_params.T)), alpha=0.1)
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_ylabel('Magnitude (dB)')

                
            plt.savefig(f'temp/pr_bnn_geo_params_s11_abs_{idx}.png')
            plt.close()
            idx += 1

if __name__ == "__main__":
    pr_bnn_geo_params()    