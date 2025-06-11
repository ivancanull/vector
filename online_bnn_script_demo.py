import argparse

from matplotlib.style import available
from numpy import s_
from src import *

from torch.utils.data import random_split
from torch import nn, optim
import matplotlib.pyplot as plt

def uncertainty_aware_resample_dataset(freqs, port_combinations, dataset, net, regularizer, sample_method, device, args, compare: bool = False, sample_size=50, sample_num=20):
    """
    Resample the dataset based on the uncertainty of the model predictions.
    This function calculates the uncertainty of the model predictions and resamples the dataset based on the uncertainty.
    Args:
    - dataset: The dataset to resample.
    - net: The Bayesian Neural Network model.
    - regularizer: The regularizer to use for the model.
    - device: The device to use for the model.
    - args: The arguments for the model.
    - compare: If True, compare the uncertainty with the true values.
    - sample_size: The number of samples to resample.
    - sample_num: The number of samples to generate for the frequency domain.

    Returns:
    - None
    """

    uncertainties = []
    uncertainties_sum = []

    if compare:
        mse_losses = []
    previous_idx = dataset.get_sample_idx()
    dataset.set_sample_idx(dataset.available_idx)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    #  sample the dataset geometric parameters 
    if sample_method in ["uniform", "random"]:
        sample_idx = np.random.choice(len(dataset.available_idx), size=sample_size, replace=False)
    elif sample_method == "uncertainty_aware":
        
        for x, y in dataloader:
            output_samples = validate(x, y, net, regularizer, False, args.samples, device) # shape: (batch_size, n_samples, freqs, pc) and not freeze

            ### Use the output variance of all ports and frequencies to calculate the uncertainty
            # calculate the variance of the absolute value of the complex number
            output_samples["ri"] = output_samples["r"] + 1j * output_samples["i"] # shape: (batch_size, n_samples, freqs, pc)
            output_samples["abs_var"] = np.var(output_samples["r"] + 1j * output_samples["i"], axis=1)
            output_samples["abs_var_sum"] = np.sum(output_samples["abs_var"], axis=(1, 2))
            uncertainties.append(output_samples["abs_var"])
            uncertainties_sum.append(output_samples["abs_var_sum"])

            if compare:
            # calculate the absolute error     
                y_true = y["r"].cpu().detach().numpy() + 1j * y["i"].cpu().detach().numpy()# shape: (batch_size, freqs, pc)
                abs_error = np.abs(np.mean(output_samples["ri"], axis=1)) - np.abs(y_true) # shape: (batch_size, freqs, pc)
                mse_loss = np.mean(abs_error ** 2, axis=(1, 2))  # mean squared error across samples and ports
                mse_losses.append(mse_loss)
        
        uncertainties = np.concatenate(uncertainties, axis=0)
        uncertainties_sum = np.concatenate(uncertainties_sum, axis=0)
        sample_idx = np.argsort(uncertainties_sum)[::-1][:sample_size]  # get the indices of the top 
        if compare:
            mse_losses = np.concatenate(mse_losses, axis=0)
        
    else:
        raise ValueError(f"Unknown sample method: {sample_method}")

    errors = []
    sample_idx = sample_idx.tolist()  # convert to list for easier indexing

    for j in range(sample_size):
        # distribute the frequencies according to the uncertainties
        if sample_method == "uniform":
            sample_freq = uniform_distribute(freqs, sample_num)
        elif sample_method == "random":
            sample_freq = np.sort(np.random.choice(freqs, size=sample_num, replace=False))
        elif sample_method == "uncertainty_aware":
            sample_freq = np.sort(uncertainty_aware_distribute(freqs, np.sum(uncertainties[sample_idx[j]], axis=1), sample_num))
        else:
            raise ValueError(f"Unknown sample method: {sample_method}")
        
        sample_freq_idx = locate_freqs_idx(freqs, sample_freq)
        for k, pc in enumerate(port_combinations):
            init_poles = basic_poles_initialize(freqs_range=(3.5e9, 8.713e9), n_poles_real=4, n_poles_complex=8) # Pre defined values for the poles
            poles, residues, d, h = vector_fit(sample_freq, dataset.dataset[sample_idx[j]]["s_params"][pc[0], pc[1], sample_freq_idx], init_poles=init_poles, max_iter=20, tol=1e-7, atol=1e-7, rtol=1e-7)

            s_params_fit = evaluate(poles, residues, d, h, frequency_to_s(freqs))
            # calculate the l2 error
            nrmse_error = fitted_error_estimate(dataset.dataset[sample_idx[j]]["s_params"][pc[0], pc[1]], s_params_fit, method="NRMSE", norm="range")
            errors.append(nrmse_error)
            # set the s_params_fit to the dataset
            dataset.dataset[sample_idx[j]]["s_params"][pc[0], pc[1]] = s_params_fit
        
    errors = np.stack(errors, axis=0)

    dataset.remove_idx(sample_idx)  # remove the sample idx from the dataset
    dataset.set_sample_idx(previous_idx)
    dataset.extend_sample_idx(sample_idx)  # set the sample idx to the dataset and expand the dataset with the new samples
    return dataset, errors

def main():
    parser = argparse.ArgumentParser(description="Run Bayesian Neural Network demo for geophysical parameters.")
    parser.add_argument("--model_name", type=str, help="Name of the model to save.")
    parser.add_argument("--data_path", type=str, default="data/spiral_inductor_s_parameters_export", help="Path directory to save the data.")
    parser.add_argument("--port_num", type=int, default=2, help="Number of ports.")
    parser.add_argument("--epochs", type=int, default=4000, help="Number of training epochs.")
    parser.add_argument("--freqs", type=float, nargs=3, help="Frequency range as min, max, step (e.g., 0.1e9 20e9 0.03e9).")
    parser.add_argument("--start_idx", type=int, help="The index to start reading geometric parameters from the filename.")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate.")
    parser.add_argument("--update_method", type=str, default="uniform", help="Method to update the dataset with new samples.")
    parser.add_argument("--retrain", action="store_true", help="Force to ratrain the model.")


    args = parser.parse_args()
    freqs = np.arange(*args.freqs)
    port_combinations = [(0, 0), (0, 1)]
    iter_num = 10  # number of iterations for training

    # initialize dataset
    dataset = SPOnlineDataset(
        args.data_path, freqs, args.port_num, args.start_idx, port_combinations, "ri"
    )

    test_idx_path = f"models/{args.model_name}_test_idx.npy"

    if os.path.exists(test_idx_path) and not args.retrain:
        print("Loading pre-saved test indices...")
        test_idx = np.load(test_idx_path)
    else:
        # split train / test
        dataset_size = len(dataset.available_idx)
        test_size = int(0.2 * dataset_size)
        test_idx = np.random.choice(dataset.available_idx, size=test_size, replace=False)
        np.save(test_idx_path, test_idx)  # save the test indices for later use

    dataset.remove_idx(test_idx.tolist())  # remove the test idx from the dataset 

    # the initial sample size
    sample_size = 50
    
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
    model_path = f"models/online_{args.model_name}_{args.update_method}_final.pth"
    save_path = f"models/online_{args.model_name}_{args.update_method}_temp.pth"

    net.to(device)
    
    if os.path.exists(model_path) and not args.retrain:
        print("Loading pre-trained model...")
        net.load_state_dict(torch.load(model_path))
        net.to(device)
        net.eval()
        sample_num = 20
        with torch.no_grad():
            dataset, errors = uncertainty_aware_resample_dataset(freqs, port_combinations, dataset, net, regularizer, args.update_method, device, args, compare=True, sample_size=sample_size, sample_num=sample_num)
            mean_error = np.mean(errors)
        print(f"Mean error of the uncertainty-aware resampling: {mean_error:.4e}")
    else:
        if os.path.exists(save_path) and not args.retrain:
            print("Loading pre-saved training state...")
            save = torch.load(save_path)
            net.load_state_dict(save["model_state_dict"])
            optimizer.load_state_dict(save["optimizer_state_dict"])
            current_iter = save["current_iter"]
            sample_idx = save["sample_idx"]
            available_idx = save["available_idx"]
            dataset.set_sample_idx(sample_idx)
            dataset.set_available_idx(available_idx)
        else:
            current_iter = 0

        for i in range(current_iter, iter_num):

            sample_num = 20
            if i == 0:
                dataset.random_sample_idx(sample_size)
            else:
                dataset, errors = uncertainty_aware_resample_dataset(freqs, port_combinations, dataset, net, regularizer, args.update_method, device, args, compare=False, sample_size=sample_size, sample_num=sample_num)
        
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            net, optimizer = train(
                epochs=args.epochs * (i + 1),
                dataloader=dataloader,
                model=net,
                optimizer=optimizer,
                data_loss_fns=data_loss_fns,
                data_loss_weights=data_loss_weights,
                bayesian_loss_fn=bayesian_loss_fn,
                bayesian_loss_weight=bayesian_loss_weight,
                regularizer=regularizer,
                device=device
            )
            torch.save(
                {"model_state_dict": net.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "current_iter": i + 1,
                 "sample_idx": dataset.get_sample_idx(),
                 "available_idx": dataset.get_available_idx()},
                save_path
            )
        torch.save(net.state_dict(), model_path)
    return

    

    net.eval()
    net.to(device)
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
            
            if idx >= 10:
                continue
            for i, pc in enumerate(port_combinations):
                init_poles = basic_poles_initialize(freqs_range=(3.5e9, 8.713e9), n_poles_real=4, n_poles_complex=8)
                
                poles, residues, d, h = vector_fit(sample_freqs,  y_true["ri"][sample_idx, i], init_poles=init_poles, max_iter=20, tol=1e-8, atol=1e-8, rtol=1e-8)
                s_params_fit = evaluate(poles, residues, d, h, frequency_to_s(freqs))
                poles_rand, residues_rand, d_rand, h_rand = vector_fit(random_freqs,  y_true["ri"][random_idx, i], init_poles=init_poles, max_iter=20, tol=1e-8, atol=1e-8, rtol=1e-8)
                s_params_fit_rand = evaluate(poles_rand, residues_rand, d_rand, h_rand, frequency_to_s(freqs))

                # calculate the l2 error
                error = fitted_error_estimate(y_true["ri"][:, i], s_params_fit, method="NRMSE", norm="range")
                print(f"Port {pc}, L2 Error (Uncertainty Aware): {error:.4e}")
                error_rand = fitted_error_estimate(y_true["ri"][:, i], s_params_fit_rand, method="NRMSE", norm="range")
                print(f"Port {pc}, L2 Error (Random): {error_rand:.4e}")
            
                print(f"Ploting Figure {idx}")
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
    return

if __name__ == "__main__":
    main()