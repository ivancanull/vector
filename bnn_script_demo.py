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

            if idx >= 10:
                continue
            for i, pc in enumerate(port_combinations):
                print(f"Ploting Figure {idx}")
                fig, axes = plt.subplots(3, 1, figsize=(12, 9))
                axes[0].plot(freqs,  y_true["r"][:, i], label='True S11 real', color='blue')
                axes[0].plot(freqs,  y_samp["r"][:, :, i].T, alpha=0.1)
                axes[0].set_xlabel('Frequency (Hz)')
                axes[0].set_ylabel('Real Part')
                axes[0].legend()

                axes[1].plot(freqs, y_true["i"][:, i], label='True S11 imag', color='blue')
                axes[1].plot(freqs, y_samp["i"][:, :, i].T, alpha=0.1)
                axes[1].set_xlabel('Frequency (Hz)')
                axes[1].set_ylabel('Imaginary Part')
                axes[1].legend()
                
                y_samp_abs = y_samp["r"] + 1j * y_samp["i"]
                y_true_abs = y_true["r"] + 1j * y_true["i"]
                axes[2].plot(freqs, np.abs(y_true_abs[:, i]), label='True S11 Abs', color='blue')
                axes[2].plot(freqs, np.abs(y_samp_abs[:, :, i].T), alpha=0.1)
                axes[2].set_xlabel('Frequency (Hz)')
                axes[2].set_ylabel('Magnitude (dB)')

                
                plt.savefig(f'temp/{args.model_name}_bnn_geo_params_s11_abs_{idx}_{str(pc)}.png')
                plt.close()
                idx += 1
    return



if __name__ == "__main__":
    main()