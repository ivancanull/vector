from src import *
import yaml
from omegaconf import DictConfig

from torch.utils.data import random_split
from torch import nn, optim
import matplotlib.pyplot as plt

# TCNS = {
#     "kernel_size": [32, 36, 32],
#     "stride": [4, 4, 4],
#     "padding": [0, 1, 0],
#     "dilation": [1, 1, 1],
#     "output_padding": [0, 1, 0]
# }

TCNS = {
    "kernel_size": [32, 32, 32],
    "stride": [3, 3, 3],
    "padding": [0, 1, 0],
    "dilation": [1, 1, 1],
    "output_padding": [0, 1, 0]
}

def load_config():
    parser = argparse.ArgumentParser(description="Run Bayesian Neural Network demo for geophysical parameters.")
    parser.add_argument("--config", type=str, default="configs/filter_tcn_base.yaml", help="Path to the configuration file.")
    parser.add_argument("--retrain", action="store_true", help="Force to ratrain the model.")

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config = DictConfig(config)
    config.retrain = args.retrain
    return config

def main():
    config = load_config()

    freqs = np.arange(config.data.freqs.min, config.data.freqs.max, config.data.freqs.step)

    if config.data.load_from_online_data:
        test_idx_path = f"models/{config.target.name}_test_idx.npy"
        print("Loading pre-saved test indices...")
        test_idx = np.load(test_idx_path)
        
        train_dataset = SPOnlineDataset(config.data.path, freqs, config.target.port_num, config.data.start_idx, config.target.port_combinations, config.data.mode)
        train_dataset.load(config.data.load_data_path)
        train_dataset.set_sample_idx(train_dataset.history_idx)

        test_dataset = SPOnlineDataset(config.data.path, freqs, config.target.port_num, config.data.start_idx, config.target.port_combinations, config.data.mode)
        test_dataset.load(config.data.load_data_path)
        test_dataset.set_sample_idx(test_idx)

    else:
        dataset = SPDataset(config.data.path, freqs, config.target.port_num, config.data.start_idx, config.target.port_combinations, config.data.mode)
        
        # split train / test
        train_size = int(0.8 * len(dataset))
        # train_size = 50
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
        test_dataset.random_sample = False  # disable random sampling for test dataset

    # create dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # model_path = f"models/{config.model.name}_{config.target.name}.pth"
    model_path = f"models/{config.case_name}.pth"

    prior_mus = [0.0] * len(config.model.hidden_dims)
    prior_sigmas = [0.1] * len(config.model.hidden_dims)
    
    if config.model.name == "mlp":
        net = MultiOutputMLP(
            input_dim=config.model.input_dim,
            hidden_dims=config.model.hidden_dims,
            output_channels=config.model.output_channels,
        )
        bayesian_loss_fn = None
        bayesian_loss_weight = 0.0
    
    elif config.model.name == "tcn":
        net = MultiOutputTCN(
            input_dim=config.model.input_dim,
            hidden_dims=config.model.hidden_dims,
            output_channels=config.model.output_channels,
            tcns=config.model.tcns,
        )
        bayesian_loss_fn = None
        bayesian_loss_weight = 0.0

    elif config.model.name == "bnn":
        net = MultiOutputBNN(
            input_dim=config.model.input_dim,
            hidden_dims=config.model.hidden_dims,
            prior_mus=prior_mus,
            prior_sigmas=prior_sigmas,
            output_channels=config.model.output_channels,
        )
        bayesian_loss_fn = BKLLoss(reduction='mean', last_layer_only=False)
        bayesian_loss_weight = 0.01
    else:
        raise ValueError("Unsupported model type. Choose 'bnn', 'tcn', or 'mlp'.")

    data_loss_fns = {
        "r": nn.MSELoss(),
        "i": nn.MSELoss()
    }

    data_loss_weights = {
        "r": 1.0,
        "i": 1.0
    }

    # x_regularizer = MinMaxRegularizer([50, 1.0, 10, 0.1e9], [70, 2.0, 20, 19.99e9])
    # x_regularizer = MinMaxRegularizer([7.7, 8.0, 7.8, 0.9, 0.7, 0.9, 0.16, 0.6, 3.5e9], [7.9, 8.2, 8.0, 1.1, 0.9, 1.1, 0.13, 0.55, 8.7e9])
    x_regularizer = MinMaxRegularizer(config.data.regularization.min, config.data.regularization.max)

    if "y_regularization" in config.data:
        y_regularizer = {
            "r": DivideRegularizer(config.data.y_regularization.r),
            "i": DivideRegularizer(config.data.y_regularization.i)
        }
    else:
        y_regularizer = {
            "r": NoneRegularizer(),
            "i": NoneRegularizer()
        }
    regularizer = (x_regularizer, y_regularizer)
    
    optimizer = optim.Adam(net.parameters(), lr=.0001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_path) and not config.retrain:
        print("Loading pre-trained model...")
        net.load_state_dict(torch.load(model_path))
    else:
        net, _ = train(
            epochs=config.train.epochs,
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
    
    # freeze

    with torch.no_grad():
        idx = 0
    
        error_metrics = ["MSE", "RMSE", "R-squared", "PSNR"]
        error_statistics = { error: [] for error in error_metrics }

        for x, y in test_loader:
            output_samples = validate(x, y, net, regularizer, True, config.test.samples, device) # shape: (batch_size, freqs, pc)
            # calculate the variance of the absolute value of the complex number
            y["r"] = y["r"].cpu().detach().numpy() # shape: (batch_size, freqs, pc)
            y["i"] = y["i"].cpu().detach().numpy() # shape: (batch_size, freqs, pc)

            if config.model.name == "tcn":
                output_samples["r"] = output_samples["r"].transpose(0, 2, 1) # shape: (batch_size, freqs, pc)
                output_samples["i"] = output_samples["i"].transpose(0, 2, 1) # shape: (batch_size, freqs, pc)
                y["r"] = y["r"].transpose(0, 2, 1) # shape: (batch_size, freqs, pc)
                y["i"] = y["i"].transpose(0, 2, 1) # shape: (batch_size, freqs, pc)

            output_samples["ri"] = output_samples["r"] + 1j * output_samples["i"] # shape: (batch_size, freqs, pc)
            y["ri"] = y["r"] + 1j * y["i"] # shape: (batch_size, freqs, pc)

            y["ri_dB"] = 20 * np.log10(np.abs(y["ri"]))
            output_samples["ri_dB"] = 20 * np.log10(np.abs(output_samples["ri"]))

            for metric in error_metrics:
                error = fitted_error_estimate(y["ri_dB"], output_samples["ri_dB"], method=metric, norm="range")
                error_statistics[metric].append(error)

            if idx < 10:
                for i, pc in enumerate(config.target.port_combinations):                
                    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
                    axes[0].plot(freqs, y["r"][0, :, i], label='True real', color='blue')
                    axes[0].plot(freqs, output_samples["r"][0, :, i], label='Predicted real', linestyle='dashed', color='orange')
                    axes[0].set_xlabel('Frequency (Hz)')
                    axes[0].set_ylabel('Real Part')
                    axes[0].legend()
                    axes[1].plot(freqs, y["i"][0, :, i], label='True imag', color='blue')
                    axes[1].plot(freqs, output_samples["i"][0, :, i], label='Predicted imag', linestyle='dashed', color='orange')
                    axes[1].set_xlabel('Frequency (Hz)')
                    axes[1].set_ylabel('Imaginary Part')
                    axes[1].legend()
                    axes[2].plot(freqs, np.abs(y["ri"])[0, :, i], label='True Abs', color='blue')
                    axes[2].plot(freqs, np.abs(output_samples["ri"])[0, :, i], label='Predicted Abs', linestyle='dashed', color='orange')
                    axes[2].set_xlabel('Frequency (Hz)')
                    axes[2].set_ylabel('Magnitude')
                    plt.savefig(f'final/{config.case_name}_{idx}_{pc}.png')
                    plt.close()
            idx += 1

        for metric in error_metrics:
            error_statistics[metric] = np.mean(error_statistics[metric])
            print(f"Test Error ({metric}): {error_statistics[metric]:.4e}")    
    
    # unfreeze
    # sample_num = 20
    # with torch.no_grad():
    #     idx = 0
    #     mse_losses = []
    #     uncertainties = []
    #     for x, y in test_loader:
    #         output_samples = validate(x, y, net, regularizer, False, config.test.samples, device) # shape: (batch_size, n_samples, freqs, pc)
    #         # calculate the variance of the absolute value of the complex number
    #         output_samples["ri"] = output_samples["r"] + 1j * output_samples["i"] # shape: (batch_size, n_samples, freqs, pc)

    #         # calculate the variance of the absolute value of the complex number
    #         y["r"] = y["r"].cpu().detach().numpy() # shape: (batch_size, freqs, pc)
    #         y["i"] = y["i"].cpu().detach().numpy() # shape: (batch_size, freqs, pc)

    #         # if config.model.name == "tcn":
    #         #     print(output_samples["r"].shape)
    #         #     output_samples["r"] = output_samples["r"].transpose(0, 2, 1) # shape: (batch_size, freqs, pc)
    #         #     output_samples["i"] = output_samples["i"].transpose(0, 2, 1) # shape: (batch_size, freqs, pc)
    #         #     y["r"] = y["r"].transpose(0, 2, 1) # shape: (batch_size, freqs, pc)
    #         #     y["i"] = y["i"].transpose(0, 2, 1) # shape: (batch_size, freqs, pc)

    #         output_samples["ri"] = output_samples["r"] + 1j * output_samples["i"] # shape: (batch_size, freqs, pc)
    #         y["ri"] = y["r"] + 1j * y["i"] # shape: (batch_size, freqs, pc)
    #         print(y["ri"].shape,)
    #         output_samples["abs_var"] = np.sum(np.var(np.abs(output_samples["ri"]), axis=1)[0], axis=-1)  # shape: (batch_size, freqs, pc)
            
    #         sample_freqs = electro_grav_sampling(freqs, sample_num, output_samples["abs_var"], lambda_gravity=0.3,
    #                     physics_model='coulomb', max_iter=10, 
    #                     init_strategy='uniform', verbose=True)
    #         sample_freqs = partition_by_uncertainty(freqs, output_samples["abs_var"], sample_num )
    #         sample_freqs = uncertainty_aware_distribute(freqs, output_samples["abs_var"], sample_num, lambda_param=0.3)
    #         print(f"Sampled Frequencies: {sample_freqs}")
    #         # locate the freqs indices
    #         sample_freqs = np.sort(sample_freqs)
    #         sample_idx = locate_freqs_idx(freqs, sample_freqs)
    #         random_idx = np.sort(np.random.choice(len(freqs), size=sample_num, replace=False))
    #         random_freqs = freqs[random_idx]
    #         random_freqs = uniform_distribute(freqs, sample_num)
    #         random_idx = locate_freqs_idx(freqs, random_freqs)
    #         uncertainty_error = 0.0
    #         random_error = 0.0
    #         s_params_fits = []
    #         s_params_fits_rand = []
    #         for i, pc in enumerate(config.target.port_combinations):
                
    #             init_poles = basic_poles_initialize(freqs_range=(3.5e9, 8.713e9), n_poles_real=4, n_poles_complex=8)
                
    #             poles, residues, d, h = vector_fit(sample_freqs,  y["ri"][0, sample_idx, i], init_poles=init_poles, max_iter=20, tol=1e-8, atol=1e-8, rtol=1e-8)
    #             s_params_fit = evaluate(poles, residues, d, h, frequency_to_s(freqs))
    #             poles_rand, residues_rand, d_rand, h_rand = vector_fit(random_freqs,  y["ri"][0, random_idx, i], init_poles=init_poles, max_iter=20, tol=1e-8, atol=1e-8, rtol=1e-8)
    #             s_params_fit_rand = evaluate(poles_rand, residues_rand, d_rand, h_rand, frequency_to_s(freqs))
    #             s_params_fits.append(s_params_fit)
    #             s_params_fits_rand.append(s_params_fit_rand)

    #             # calculate the l2 error
    #             uncertainty_error += fitted_error_estimate(y["ri"][0, :, i], s_params_fit, method="MReE")
    #             random_error += fitted_error_estimate(y["ri"][0, :, i], s_params_fit_rand, method="MReE")
    #             print("Port Combination:", pc)
    #             print(f"Uncertainty Aware Error: {uncertainty_error:.4e}")
    #             print(f"Random Error: {random_error:.4e}")
            

    #         uncertainty_error /= len(config.target.port_combinations)
    #         random_error /= len(config.target.port_combinations)
    #         total_uncertainty_error += uncertainty_error
    #         total_random_error += random_error

    #         if idx < 10:
    #             for i, pc in enumerate(config.target.port_combinations):
    #                 print(f"Ploting Figure {idx}")
    #                 fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    #                 axes[0].plot(freqs, y["r"][0, :, i], label='True real', color='blue')
    #                 axes[0].plot(freqs, output_samples["r"][0, :, :, i].T, label='Predicted real', linestyle='dashed', color='orange', alpha=0.1)
    #                 axes[0].plot(freqs, np.real(s_params_fits[i]), label='Fitted real', linestyle='dashed', color='orange')
    #                 axes[0].plot(freqs, np.real(s_params_fits_rand[i]), label='Fitted real (random)', linestyle='dotted', color='green')
    #                 axes[0].set_xlabel('Frequency (Hz)')
    #                 axes[0].set_ylabel('Real Part')
    #                 axes[0].legend()
    #                 axes[1].plot(freqs, y["i"][0, :, i], label='True imag', color='blue')
    #                 axes[1].plot(freqs, output_samples["i"][0, :, :, i].T, label='Predicted imag', linestyle='dashed', color='orange', alpha=0.1)
    #                 axes[1].plot(freqs, np.imag(s_params_fits[i]), label='Fitted imag', linestyle='dashed', color='orange')
    #                 axes[1].plot(freqs, np.imag(s_params_fits_rand[i]), label='Fitted imag (random)', linestyle='dotted', color='green')
    #                 axes[1].set_xlabel('Frequency (Hz)')
    #                 axes[1].set_ylabel('Imaginary Part')
    #                 axes[1].legend()
    #                 axes[2].plot(freqs, np.abs(y["ri"])[0, :, i], label='True Abs', color='blue')
    #                 axes[2].plot(freqs, np.abs(output_samples["ri"])[0,:,  :, i].T, label='Predicted Abs', linestyle='dashed', color='orange', alpha=0.1)
    #                 axes[2].set_xlabel('Frequency (Hz)')
    #                 axes[2].set_ylabel('Magnitude')

    #                 axes[3].plot(freqs, output_samples["abs_var"], label='Absolute Variance')
    #                 axes[3].scatter(sample_freqs, output_samples["abs_var"][sample_idx], color='red', label='Sampled Variance', marker='x')
    #                 axes[3].set_xlabel('Frequency (Hz)')
    #                 axes[3].set_ylabel('Variance (dB)')

    #                 plt.savefig(f'final/{config.case_name}_{idx}_{pc}.png')
    #                 plt.close()
                
    #         idx += 1

            
            
            
    #         continue
    #         sample_freqs = uncertainty_aware_distribute(freqs, y_samp["abs_var"], sample_num)

    #         locate the freqs indices
    #         sample_freqs = np.sort(sample_freqs)
    #         sample_idx = locate_freqs_idx(freqs, sample_freqs)
    #         random_idx = np.sort(np.random.choice(len(freqs), size=sample_num, replace=False))
    #         random_freqs = freqs[random_idx]
            
    #         uncertainty_error = 0.0
    #         random_error = 0.0

    #         for i, pc in enumerate(port_combinations):
                
    #             init_poles = basic_poles_initialize(freqs_range=(3.5e9, 8.713e9), n_poles_real=4, n_poles_complex=8)
                
    #             poles, residues, d, h = vector_fit(sample_freqs,  y_true["ri"][sample_idx, i], init_poles=init_poles, max_iter=20, tol=1e-8, atol=1e-8, rtol=1e-8)
    #             s_params_fit = evaluate(poles, residues, d, h, frequency_to_s(freqs))
    #             poles_rand, residues_rand, d_rand, h_rand = vector_fit(random_freqs,  y_true["ri"][random_idx, i], init_poles=init_poles, max_iter=20, tol=1e-8, atol=1e-8, rtol=1e-8)
    #             s_params_fit_rand = evaluate(poles_rand, residues_rand, d_rand, h_rand, frequency_to_s(freqs))

    #             calculate the l2 error
    #             uncertainty_error += fitted_error_estimate(y_true["ri"][:, i], s_params_fit, method="MReE")
    #             random_error += fitted_error_estimate(y_true["ri"][:, i], s_params_fit_rand, method="MReE")
    #             print("Port Combination:", pc)
    #             print(f"Uncertainty Aware Error: {uncertainty_error:.4e}")
    #             print(f"Random Error: {random_error:.4e}")

    #         uncertainty_error /= len(port_combinations)
    #         random_error /= len(port_combinations)
    #         total_uncertainty_error += uncertainty_error
    #         total_random_error += random_error
            
    #         if idx < 10:
            
    #             print(f"Ploting Figure {idx}")
    #             fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    #             axes[0].plot(freqs, y_true["r"][:, i], label='True real', color='blue')
    #             axes[0].plot(freqs, y_samp["r"][:, :, i].T, alpha=0.1)
    #             axes[0].plot(freqs, np.real(s_params_fit), label='Fitted real', linestyle='dashed', color='orange')
    #             axes[0].plot(freqs, np.real(s_params_fit_rand), label='Fitted real (random)', linestyle='dotted', color='green')
    #             axes[0].set_xlabel('Frequency (Hz)')
    #             axes[0].set_ylabel('Real Part')
    #             axes[0].legend()

    #             axes[1].plot(freqs, y_true["i"][:, i], label='True imag', color='blue')
    #             axes[1].plot(freqs, y_samp["i"][:, :, i].T, alpha=0.1)
    #             axes[1].plot(freqs, np.imag(s_params_fit), label='Fitted imag', linestyle='dashed', color='orange')
    #             axes[1].plot(freqs, np.imag(s_params_fit_rand), label='Fitted imag (random)', linestyle='dotted', color='green')
    #             axes[1].set_xlabel('Frequency (Hz)')
    #             axes[1].set_ylabel('Imaginary Part')
    #             axes[1].legend()
                
    #             y_samp_abs = y_samp["r"] + 1j * y_samp["i"]
    #             y_true_abs = y_true["r"] + 1j * y_true["i"]
    #             axes[2].plot(freqs, np.abs(y_true_abs[:, i]), label='True Abs', color='blue')
    #             axes[2].plot(freqs, np.abs(y_samp_abs[:, :, i].T), alpha=0.1)
    #             axes[2].plot(freqs, np.abs(s_params_fit), label='Fitted Abs', linestyle='dashed', color='orange')
    #             axes[2].plot(freqs, np.abs(s_params_fit_rand), label='Fitted Abs (random)', linestyle='dotted', color='green')
    #             axes[2].set_xlabel('Frequency (Hz)')
    #             axes[2].set_ylabel('Magnitude (dB)')

    #             axes[3].plot(freqs, y_samp["abs_var"], label='Absolute Variance')
    #             axes[3].scatter(sample_freqs, y_samp["abs_var"][sample_idx], color='red', label='Sampled Variance', marker='x')
    #             axes[3].set_xlabel('Frequency (Hz)')
    #             axes[3].set_ylabel('Variance (dB)')

    #             plt.savefig(f'temp/{args.model_name}_bnn_geo_params_abs_{idx}_{str(pc)}.png')
    #             plt.close()
            
    #         idx += 1

            
    #     print(f"Total Uncertainty Aware Error: {total_uncertainty_error:.4e}")
    #     print(f"Total Random Error: {total_random_error:.4e}")
    #     print(f"Average Uncertainty Aware Error: {total_uncertainty_error / idx:.4e}")
    #     print(f"Average Random Error: {total_random_error / idx:.4e}")
    # return

if __name__ == "__main__":
    main()