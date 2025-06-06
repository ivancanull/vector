import argparse
from src import *

from torch.utils.data import random_split
from torch import nn, optim
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Run Bayesian Neural Network demo for geophysical parameters.")
    parser.add_argument("--model_name", type=str, help="Name of the model to save.")
    parser.add_argument("--data_path", type=str, default="data/spiral_inductor_s_parameters_export", help="Path directory to save the data.")
    parser.add_argument("--port_num", type=int, default=2, help="Number of ports.")
    parser.add_argument("--epochs", type=int, default=4000, help="Number of training epochs.")
    parser.add_argument("--freqs", type=float, nargs=3, help="Frequency range as min, max, step (e.g., 0.1e9 20e9 0.03e9).")
    parser.add_argument("--start_idx", type=int, help="The index to start reading geometric parameters from the filename.")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate.")
    parser.add_argument("--retrain", action="store_true", help="Force to ratrain the model.")


    args = parser.parse_args()
    freqs = np.arange(*args.freqs)
    port_combinations = [(0, 0), (0, 1)]
    dataset = SPDataset(
        args.data_path, freqs, args.port_num, args.start_idx, port_combinations, "ri"
    )
    
    # split train / test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    test_dataset.random_sample = False  # disable random sampling for test dataset

    # create dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_path = f"models/{args.model_name}.pth"
    
    input_dim = 9
    hidden_dims = [256, 256, 256]
    prior_mus = [0.0] * len(hidden_dims)
    prior_sigmas = [0.3] * len(hidden_dims)
    output_channels = {
        'r': (2, None),   
        'i': (2, None),
    }
    
    # Create model
    net = MultiOutputBNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        prior_mus=prior_mus,
        prior_sigmas=prior_sigmas,
        output_channels=output_channels,

    )

    data_loss_fns = {
        "r": nn.MSELoss(),
        "i": nn.MSELoss()
    }

    data_loss_weights = {
        "r": 1.0,
        "i": 1.0
    }

    # x_regularizer = MinMaxRegularizer([50, 1.0, 10, 0.1e9], [70, 2.0, 20, 19.99e9])
    x_regularizer = MinMaxRegularizer([7.7, 8.0, 7.8, 0.9, 0.7, 0.9, 0.16, 0.6, 3.5e9], [7.9, 8.2, 8.0, 1.1, 0.9, 1.1, 0.13, 0.55, 8.7e9])
    y_regularizer = {
        "r": NoneRegularizer(),
        "i": NoneRegularizer()
    }
    regularizer = (x_regularizer, y_regularizer)
    bayesian_loss_fn = BKLLoss(reduction='mean', last_layer_only=False)
    bayesian_loss_weight = 0.01

    optimizer = optim.Adam(net.parameters(), lr=.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_path) and not args.retrain:
        print("Loading pre-trained model...")
        net.load_state_dict(torch.load(model_path))
    else:
        net = train(
            epochs=args.epochs,
            dataloader=train_loader,
            model=net,
            optimizer=optimizer,
            data_loss_fns=data_loss_fns,
            data_loss_weights=data_loss_weights,
            bayesian_loss_fn=bayesian_loss_fn,
            bayesian_loss_weight=bayesian_loss_weight,
            regularizer=regularizer,
            device=device
        )
        torch.save(net.state_dict(), model_path)
    
    net.eval()
    net.to(device)
    total_uncertainty_error = 0.0
    total_random_error = 0.0

    with torch.no_grad():
        idx = 0
        for x, y in test_loader:
            x = x.to(device)
            x = x_regularizer.regularize(x)
            
            y_true = {
                "r": y["r"].cpu().detach().numpy()[0],
                "i": y["i"].cpu().detach().numpy()[0],
            }
            y_samp = {
                "r": [],
                "i": [],
            }
            for s in range(args.samples):
                pred = net(x)
                # print(pred["r"].cpu().detach().numpy().shape) # (1, freqs, 2)
                y_samp["r"].append(pred["r"].cpu().detach().numpy()) 
                y_samp["i"].append(pred["i"].cpu().detach().numpy())

            y_samp["r"] = np.concatenate(y_samp["r"], axis=0)
            y_samp["i"] = np.concatenate(y_samp["i"], axis=0)

           
            # calculate the s11 absolute dB value
            y_samp["ri"] = y_samp["r"] + 1j * y_samp["i"]
            y_true["ri"] = y_true["r"] + 1j * y_true["i"]
            y_samp["abs"] = np.abs(y_samp["ri"]) 
            y_true["abs"] = np.abs(y_true["ri"])

            y_samp["abs_dB"] = 20 * np.log10(y_samp["abs"])
            y_true["abs_dB"] = 20 * np.log10(y_true["abs"])

            # calculate the variance of the abs_dB 
            y_samp["abs_var"] = np.sum(np.var(y_samp["abs"], axis=0), axis=1)
            
            sample_num = 20
            sample_freqs = uncertainty_aware_distribute(freqs, y_samp["abs_var"], sample_num)

            # locate the freqs indices
            sample_freqs = np.sort(sample_freqs)
            sample_idx = locate_freqs_idx(freqs, sample_freqs)
            random_idx = np.sort(np.random.choice(len(freqs), size=sample_num, replace=False))
            random_freqs = freqs[random_idx]
            
            uncertainty_error = 0.0
            random_error = 0.0

            for i, pc in enumerate(port_combinations):
                
                init_poles = basic_poles_initialize(freqs_range=(3.5e9, 8.713e9), n_poles_real=4, n_poles_complex=8)
                
                poles, residues, d, h = vector_fit(sample_freqs,  y_true["ri"][sample_idx, i], init_poles=init_poles, max_iter=20, tol=1e-8, atol=1e-8, rtol=1e-8)
                s_params_fit = evaluate(poles, residues, d, h, frequency_to_s(freqs))
                poles_rand, residues_rand, d_rand, h_rand = vector_fit(random_freqs,  y_true["ri"][random_idx, i], init_poles=init_poles, max_iter=20, tol=1e-8, atol=1e-8, rtol=1e-8)
                s_params_fit_rand = evaluate(poles_rand, residues_rand, d_rand, h_rand, frequency_to_s(freqs))

                # calculate the l2 error
                uncertainty_error += fitted_error_estimate(y_true["ri"][:, i], s_params_fit, method="MReE")
                random_error += fitted_error_estimate(y_true["ri"][:, i], s_params_fit_rand, method="MReE")
                # print("Port Combination:", pc)
                # print(f"Uncertainty Aware Error: {uncertainty_error:.4e}")
                # print(f"Random Error: {random_error:.4e}")

            uncertainty_error /= len(port_combinations)
            random_error /= len(port_combinations)
            total_uncertainty_error += uncertainty_error
            total_random_error += random_error
            
            if idx < 10:
            
                # print(f"Ploting Figure {idx}")
                fig, axes = plt.subplots(4, 1, figsize=(12, 12))
                axes[0].plot(freqs, y_true["r"][:, i], label='True real', color='blue')
                axes[0].plot(freqs, y_samp["r"][:, :, i].T, alpha=0.1)
                axes[0].plot(freqs, np.real(s_params_fit), label='Fitted real', linestyle='dashed', color='orange')
                axes[0].plot(freqs, np.real(s_params_fit_rand), label='Fitted real (random)', linestyle='dotted', color='green')
                axes[0].set_xlabel('Frequency (Hz)')
                axes[0].set_ylabel('Real Part')
                axes[0].legend()

                axes[1].plot(freqs, y_true["i"][:, i], label='True imag', color='blue')
                axes[1].plot(freqs, y_samp["i"][:, :, i].T, alpha=0.1)
                axes[1].plot(freqs, np.imag(s_params_fit), label='Fitted imag', linestyle='dashed', color='orange')
                axes[1].plot(freqs, np.imag(s_params_fit_rand), label='Fitted imag (random)', linestyle='dotted', color='green')
                axes[1].set_xlabel('Frequency (Hz)')
                axes[1].set_ylabel('Imaginary Part')
                axes[1].legend()
                
                y_samp_abs = y_samp["r"] + 1j * y_samp["i"]
                y_true_abs = y_true["r"] + 1j * y_true["i"]
                axes[2].plot(freqs, np.abs(y_true_abs[:, i]), label='True Abs', color='blue')
                axes[2].plot(freqs, np.abs(y_samp_abs[:, :, i].T), alpha=0.1)
                axes[2].plot(freqs, np.abs(s_params_fit), label='Fitted Abs', linestyle='dashed', color='orange')
                axes[2].plot(freqs, np.abs(s_params_fit_rand), label='Fitted Abs (random)', linestyle='dotted', color='green')
                axes[2].set_xlabel('Frequency (Hz)')
                axes[2].set_ylabel('Magnitude (dB)')

                axes[3].plot(freqs, y_samp["abs_var"], label='Absolute Variance')
                axes[3].scatter(sample_freqs, y_samp["abs_var"][sample_idx], color='red', label='Sampled Variance', marker='x')
                axes[3].set_xlabel('Frequency (Hz)')
                axes[3].set_ylabel('Variance (dB)')

                plt.savefig(f'temp/{args.model_name}_bnn_geo_params_abs_{idx}_{str(pc)}.png')
                plt.close()
            
            idx += 1

            
        print(f"Total Uncertainty Aware Error: {total_uncertainty_error:.4e}")
        print(f"Total Random Error: {total_random_error:.4e}")
        print(f"Average Uncertainty Aware Error: {total_uncertainty_error / idx:.4e}")
        print(f"Average Random Error: {total_random_error / idx:.4e}")
    return

if __name__ == "__main__":
    main()