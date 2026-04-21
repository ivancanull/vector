from matplotlib.style import available
from src import *

from torch.utils.data import random_split
from torch import nn, optim
import matplotlib.pyplot as plt

def uncertainty_aware_resample_dataset(freqs, 
                                       port_combinations, 
                                       dataset: SPOnlineDataset, 
                                       net, 
                                       regularizer, 
                                       dataset_update_method, 
                                       freq_sample_method, 
                                       device, 
                                       compare: bool = False, 
                                       forget: bool = True, 
                                       n_poles_real=4,
                                       n_poles_complex=8,
                                       data_sample_size=50, 
                                       freq_sample_size=20, 
                                       test_sample_size=100, 
                                       save_name="temp"):
    """
    Resample the dataset based on the uncertainty of the model predictions.
    This function calculates the uncertainty of the model predictions and resamples the dataset based on the uncertainty.
    Args:
    - dataset: The dataset to resample.
    - net: The Bayesian Neural Network model.
    - regularizer: The regularizer to use for the model.
    - dataset_update_method: The method to use for updating the dataset. Options are "random", "uncertainty_aware".
    - freq_sample_method: The method to use for sampling the frequencies. Options are "uniform", "random", "uncertainty_aware".
    - device: The device to use for the model.
    - config: The configurations for the model.
    - compare: If True, compare the uncertainty with the true values.
    - data_sample_size: The number of samples to resample.
    - freq_sample_size: The number of samples to generate for the frequency domain.
    - test_sample_size: The number of samples to generate for the test set.
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
    if dataset_update_method == "random" and freq_sample_method != "uncertainty_aware":
        sample_idx = np.random.choice(dataset.get_sample_idx(), size=data_sample_size, replace=False)
    elif dataset_update_method in ["random", "uncertainty_aware"]:
        idx = 0
        for x, y in dataloader:
            output_samples = validate(x, y, net, regularizer, False, test_sample_size, device) # shape: (batch_size, n_samples, freqs, pc) and not freeze
            ### Use the output variance of all ports and frequencies to calculate the uncertainty
            # calculate the variance of the absolute value of the complex number
            output_samples["ri"] = output_samples["r"] + 1j * output_samples["i"] # shape: (batch_size, n_samples, freqs, pc)
            output_samples["abs_var"] = np.var(output_samples["r"] + 1j * output_samples["i"], axis=1)
            output_samples["abs_var_sum"] = np.sum(output_samples["abs_var"], axis=(1, 2))
            uncertainties.append(output_samples["abs_var"])
            uncertainties_sum.append(output_samples["abs_var_sum"])
            # plot
            if idx < 5:
                for k, pc in enumerate(port_combinations):
                    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
                    axes[0].plot(freqs, y["r"].cpu().detach().numpy()[0, :, k], label='True real', color='blue')
                    axes[0].plot(freqs, output_samples["r"][0, :, :, k].T, label='Predicted real', linestyle='dashed', color='orange', alpha=0.2)
                    axes[0].plot()
                    axes[0].set_xlabel('Frequency (Hz)')
                    axes[0].set_ylabel('Real Part')
                    axes[0].legend()
                    axes[1].plot(freqs, y["i"].cpu().detach().numpy()[0, :, k], label='True imag', color='blue')
                    axes[1].plot(freqs, output_samples["i"][0, :, :, k].T, label='Predicted imag', linestyle='dashed', color='orange', alpha=0.2)
                    axes[1].set_xlabel('Frequency (Hz)')
                    axes[1].set_ylabel('Imaginary Part')
                    axes[1].legend()
                    axes[2].plot(freqs, np.abs(y["r"].cpu().detach().numpy()[0, :, k] + 1j * y["i"].cpu().detach().numpy()[0, :, k]), label='True Abs', color='blue')
                    axes[2].plot(freqs, np.abs(output_samples["ri"][0, :, :, k].T), label='Predicted Abs', linestyle='dashed', color='orange', alpha=0.2)
                    axes[2].set_xlabel('Frequency (Hz)')
                    axes[2].set_ylabel('Magnitude')
                    plt.savefig(f'uncertainty/{save_name}_{idx}_{k}.png')
                    plt.close()
            idx += 1

            if compare:
            # calculate the absolute error     
                y_true = y["r"].cpu().detach().numpy() + 1j * y["i"].cpu().detach().numpy()# shape: (batch_size, freqs, pc)
                abs_error = np.abs(np.mean(output_samples["ri"], axis=1)) - np.abs(y_true) # shape: (batch_size, freqs, pc)
                mse_loss = np.mean(abs_error ** 2, axis=(1, 2))  # mean squared error across samples and ports
                mse_losses.append(mse_loss)

        uncertainties = np.concatenate(uncertainties, axis=0)
        uncertainties_sum = np.concatenate(uncertainties_sum, axis=0)
        if dataset_update_method == "random":
            sample_idx_ = np.random.choice(len(dataset.available_idx), size=data_sample_size, replace=False)  # get the indices of the top uncertainties
            sample_idx = np.array(dataset.available_idx)[sample_idx_]  # convert to the actual indices in the dataset
            # sample_idx = np.random.choice(dataset.available_idx, size=data_sample_size, replace=False)
        else:
            sample_idx_ = uncertainty_aware_distribute(np.arange(len(uncertainties_sum)), uncertainties_sum, data_sample_size, lambda_param=0.1)
            # sample_idx_ = np.argsort(uncertainties_sum)[::-1][:data_sample_size] # get the indices of the top
            sample_idx = np.array(dataset.get_sample_idx())[sample_idx_] # convert to the actual indices in the dataset
        if compare:
            mse_losses = np.concatenate(mse_losses, axis=0)
    else:
        raise ValueError(f"Unknown sample method: {dataset_update_method}")

    errors = []
    sample_idx = sample_idx.tolist()  # convert to list for easier indexing
    # sample freq if is not "full"
    if freq_sample_method != "full":
        for j in range(data_sample_size):
            # distribute the frequencies according to the uncertainties
            if freq_sample_method == "uniform":
                sample_freq = uniform_distribute(freqs, freq_sample_size)
            elif freq_sample_method == "random":
                sample_freq = np.sort(np.random.choice(freqs, size=freq_sample_size, replace=False))
            elif freq_sample_method == "uncertainty_aware":
                sample_freq = np.sort(partition_by_uncertainty(freqs, np.sum(uncertainties[sample_idx_[j]], axis=1), freq_sample_size))
            else:
                raise ValueError(f"Unknown sample method: {freq_sample_method}")
            sample_freq_idx = locate_freqs_idx(freqs, sample_freq)
            for k, pc in enumerate(port_combinations):
                init_poles = basic_poles_initialize(freqs_range=(freqs[0], freqs[-1]), n_poles_real=n_poles_real, n_poles_complex=n_poles_complex) # Pre defined values for the poles
                poles, residues, d, h = vector_fit(sample_freq, dataset.dataset[sample_idx[j]]["s_params"][pc[0], pc[1], sample_freq_idx], init_poles=init_poles, max_iter=30, tol=1e-7, atol=1e-7, rtol=1e-7)
                s_params_fit = evaluate(poles, residues, d, h, frequency_to_s(freqs))
                # plot
                nrmse_error = fitted_error_estimate(dataset.dataset[sample_idx[j]]["s_params"][pc[0], pc[1]], s_params_fit, method="MSE", norm="range")
                # if nrmse_error > 0.1:
                #     fig, axes = plt.subplots(3, 1, figsize=(12, 9))
                #     axes[0].plot(freqs, dataset.dataset[sample_idx[j]]["s_params"][pc[0], pc[1], :].real, label='True real', color='blue')
                #     axes[0].plot(freqs, s_params_fit.real, label='Fitted real', linestyle='dashed', color='orange')
                #     axes[0].set_xlabel('Frequency (Hz)')
                #     axes[0].set_ylabel('Real Part')
                #     axes[0].legend()
                #     axes[1].plot(freqs, dataset.dataset[sample_idx[j]]["s_params"][pc[0], pc[1], :].imag, label='True imag', color='blue')
                #     axes[1].plot(freqs, s_params_fit.imag, label='Fitted imag', linestyle='dashed', color='orange')
                #     axes[1].set_xlabel('Frequency (Hz)')
                #     axes[1].set_ylabel('Imaginary Part')
                #     axes[1].legend()
                #     axes[2].plot(freqs, 20 * np.log10(np.abs(dataset.dataset[sample_idx[j]]["s_params"][pc[0], pc[1], :])), label='True Abs', color='blue')
                #     axes[2].plot(freqs, 20 * np.log10(np.abs(s_params_fit)), label='Fitted Abs', linestyle='dashed', color='orange')
                #     axes[2].set_xlabel('Frequency (Hz)')
                #     axes[2].set_ylabel('Magnitude (dB)')
                #     plt.savefig(f'temp/{save_name}_vf_{j}_{pc}.png')
                #     plt.close()
                # calculate the l2 error
                nrmse_error = fitted_error_estimate(dataset.dataset[sample_idx[j]]["s_params"][pc[0], pc[1]], s_params_fit, method="MSE", norm="range")
                errors.append(nrmse_error)
                # set the s_params_fit to the dataset
                dataset.dataset[sample_idx[j]]["s_params"][pc[0], pc[1]] = s_params_fit
    errors = np.stack(errors, axis=0)

    dataset.remove_idx(sample_idx)  # remove the sample idx from the dataset
    dataset.extend_history_idx(sample_idx)
    if forget:
        # if forget is True, we will remove the previous sample idx from the dataset
        dataset.set_sample_idx(sample_idx)
    else:
        dataset.set_sample_idx(previous_idx)
        dataset.extend_sample_idx(sample_idx)  # set the sample idx to the dataset and expand the dataset with the new samples
    return dataset, errors

def online_validate(dataset, validate_idx, net, regularizer, device, test_sample_size):
    """
    Validate the model on the dataset and resample it based on the uncertainty of the model predictions.
    Args:
    - dataset: The dataset to resample.
    - validate_idx: The indices to validate the model on.
    - net: The Bayesian Neural Network model.
    - regularizer: The regularizer to use for the model.
    - device: The device to use for the model.
    - test_sample_size: The number of samples to generate for the test set.

    Returns:
    - None
    """
    previous_idx = dataset.get_sample_idx()
    dataset.set_sample_idx(validate_idx)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    mean_dB_relative_error = []
    with torch.no_grad():
        for x, y in dataloader:
            output_samples = validate(x, y, net, regularizer, True, test_sample_size, device)
            output_samples["ri"] = output_samples["r"] + 1j * output_samples["i"]  # shape: (batch_size, n_samples, freqs, pc)
            y["ri"] = y["r"].cpu().detach().numpy() + 1j * y["i"].cpu().detach().numpy()  # shape: (batch_size, freqs, pc)
            output_samples["dB"] = 20 * np.log10(np.abs(output_samples["ri"]))
            y["dB"] = 20 * np.log10(np.abs(y["ri"]))
            abs_error = np.abs(output_samples["dB"] - y["dB"])  # shape: (batch_size, freqs, pc)
            mean_dB_relative_error.append(np.mean(abs_error, axis=(1, 2)) / (np.max(y["dB"]) - np.min(y["dB"]))) # mean squared error across samples and ports
        
    mean_dB_relative_error = np.mean(np.concatenate(mean_dB_relative_error, axis=0))
    print(f"Mean dB relative error on validation set: {mean_dB_relative_error:.4e}")
    dataset.set_sample_idx(previous_idx)  # set the sample idx to the previous idx
    return

def main():

    config = load_config()

    freqs = np.arange(config.data.freqs.min, config.data.freqs.max, config.data.freqs.step)
    iter_num = config.iter_num  # number of iterations for training

    # initialize dataset
    dataset = SPOnlineDataset(
        config.data.path, freqs, config.target.port_num, config.data.start_idx, config.target.port_combinations, config.data.mode
    )

    test_idx_path = f"models/{config.target.name}_test_idx.npy"

    if os.path.exists(test_idx_path) and not config.resample:
        print("Loading pre-saved test indices...")
        test_idx = np.load(test_idx_path)
    else:
        # split train / test
        dataset_size = len(dataset.available_idx)
        test_size = int(0.2 * dataset_size)
        test_idx = np.random.choice(dataset.available_idx, size=test_size, replace=False)
        dataset.remove_idx(test_idx.tolist())  # remove the test idx from the dataset 
        np.save(test_idx_path, test_idx)  # save the test indices for later use

    dataset.remove_idx(test_idx.tolist())  # remove the test idx from the dataset 

    val_idx_path = f"models/{config.target.name}_val_idx.npy"
    if os.path.exists(val_idx_path) and not config.resample:
        print("Loading pre-saved validation indices...")
        val_idx = np.load(val_idx_path)
    else:
        # split train / validation
        dataset_size = len(dataset.available_idx)
        val_size = int(0.2 * dataset_size)
        val_idx = np.random.choice(dataset.available_idx, size=val_size, replace=False)
        dataset.remove_idx(val_idx.tolist())
        np.save(val_idx_path, val_idx)  # save the validation indices for later use
    
    dataset.set_sample_idx(val_idx.tolist())  # set the sample idx to the validation idx
    prior_mus = [0.0] * len(config.guide_model.hidden_dims)
    prior_sigmas = [0.1] * len(config.guide_model.hidden_dims)

    net = MultiOutputBNN(
        input_dim=config.guide_model.input_dim,
        hidden_dims=config.guide_model.hidden_dims,
        prior_mus=prior_mus,
        prior_sigmas=prior_sigmas,
        output_channels=config.guide_model.output_channels,
    )

    data_loss_fns = {
        "r": nn.MSELoss(),
        "i": nn.MSELoss()
    }

    data_loss_weights = {
        "r": 1.0,
        "i": 1.0
    }

    x_regularizer = MinMaxRegularizer(config.data.regularization.min, config.data.regularization.max)

    if hasattr(config.data, "y_regularization"):
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
    bayesian_loss_fn = BKLLoss(reduction='mean', last_layer_only=False)
    bayesian_loss_weight = 0.01

    optimizer = optim.Adam(net.parameters(), lr=.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guide_model_path = f"models/{config.save_name}_guide_final.pth"
    save_path = f"models/{config.save_name}_guide_temp.pth"
    dataset_path = f"data/{config.save_name}_dataset.pkl"

    net.to(device)
    current_iter = 0

    if os.path.exists(save_path) and not config.retrain:
        save = torch.load(save_path)
        net.load_state_dict(save["model_state_dict"])
        optimizer.load_state_dict(save["optimizer_state_dict"])
        current_iter = save["current_iter"]
        sample_idx = save["sample_idx"]
        available_idx = save["available_idx"]
        dataset.set_sample_idx(sample_idx)
        dataset.set_available_idx(available_idx)
    
    if os.path.exists(guide_model_path) and not config.retrain and current_iter >= iter_num:
        print("Loading pre-trained model...")
        # load the model from the save path
        save = torch.load(guide_model_path)
        net.load_state_dict(torch.load(guide_model_path))
        net.eval()
        # dataset.load(dataset_path)  # load the dataset from the dataset path

    else:
        if config.retrain:
            current_iter = 0

        for i in range(current_iter, iter_num):

            if i == 0:
                dataset.random_sample_idx(config.online.data_sample_size)
                dataset.extend_history_idx(dataset.get_sample_idx())  # extend the history idx with the sample idx
            else:
                dataset, errors = uncertainty_aware_resample_dataset(
                    freqs, 
                    config.target.port_combinations, 
                    dataset, 
                    net, 
                    regularizer, 
                    config.online.dataset_update_method, 
                    config.online.freq_sample_method, 
                    device, 
                    compare=False, 
                    forget=config.online.forget, 
                    n_poles_real=config.vector_fitting.n_poles_real,
                    n_poles_complex=config.vector_fitting.n_poles_complex,
                    data_sample_size=config.online.data_sample_size, 
                    freq_sample_size=config.online.freq_sample_size, 
                    test_sample_size=config.test.samples, 
                    save_name=f"{config.save_name}_{i}")
                print(f"Iteration {i + 1}/{iter_num}, Sampled {len(dataset.get_sample_idx())} samples, Errors: {np.mean(errors):.4e}")

            if config.online.dataset_update_method != "uncertainty_aware" and config.online.freq_sample_method != "uncertainty_aware":
                print("No uncertainty-aware resampling, skipping training.")
                continue
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            net, optimizer = train(
                epochs=config.train.epochs,
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
            # validate
            online_validate(dataset, val_idx, net, regularizer, device, config.test.samples)

        return
        net.unfreeze()  # unfreeze the model before saving
        torch.save(net.state_dict(), guide_model_path)
        # save the online dataset
        dataset.save(dataset_path=dataset_path)
    # test the model
    
    dataset.set_sample_idx(test_idx.tolist())  # set the available idx to the test idx

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    error_metrics = ["MSE", "RMSE", "R-squared", "PSNR"]
    error_statistics = { error: [] for error in error_metrics }

    with torch.no_grad():
        idx = 0
        for x, y in test_loader:
            output_samples = validate(x, y, net, regularizer, True, config.test.samples, device)  # shape: (batch_size, n_samples, freqs, pc)
            output_samples["ri"] = output_samples["r"] + 1j * output_samples["i"] # shape: (batch_size, freqs, pc)
            y["ri"] = y["r"].cpu().detach().numpy() + 1j * y["i"].cpu().detach().numpy() # shape: (batch_size, freqs, pc)
            output_samples["dB"] = 20 * np.log10(np.abs(output_samples["ri"]))
            y["dB"] = 20 * np.log10(np.abs(y["ri"]))
            for metric in error_metrics:
                error = fitted_error_estimate(y["ri"], output_samples["ri"], method=metric, norm="range")
                error_statistics[metric].append(error)
            if idx < 10:
                for i, pc in enumerate(config.target.port_combinations):
                    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
                    axes[0].plot(freqs, y["r"].cpu().detach().numpy()[0, :, i], label='True real', color='blue')
                    axes[0].plot(freqs, output_samples["r"][0, :, i], label='Predicted real', linestyle='dashed', color='orange')
                    axes[0].set_xlabel('Frequency (Hz)')
                    axes[0].set_ylabel('Real Part')
                    axes[0].legend()
                    axes[1].plot(freqs, y["i"].cpu().detach().numpy()[0, :, i], label='True imag', color='blue')
                    axes[1].plot(freqs, output_samples["i"][0, :, i], label='Predicted imag', linestyle='dashed', color='orange')
                    axes[1].set_xlabel('Frequency (Hz)')
                    axes[1].set_ylabel('Imaginary Part')
                    axes[1].legend()
                    axes[2].plot(freqs, y["dB"][0, :, i], label='True Abs', color='blue')
                    axes[2].plot(freqs, output_samples["dB"][0, :, i], label='Predicted Abs', linestyle='dashed', color='orange')
                    axes[2].set_xlabel('Frequency (Hz)')
                    axes[2].set_ylabel('Magnitude')
                    plt.savefig(f'final/{config.save_name}_{idx}_{pc}.png')
                    plt.close()
            idx += 1
        for metric in error_metrics:
            error_statistics[metric] = np.mean(error_statistics[metric])
            print(f"Test Error ({metric}): {error_statistics[metric]:.4e}")    

if __name__ == "__main__":
    main()