from src import *
import matplotlib.pyplot as plt

def test_pr():
    # load the dataset
    dataset_path = "/public/home/zhanghf/projects/vector/data/spiral_inductor_s_parameters_export"
    port_num = 2
    function_from_touchstone = FunctionFromTouchstone(port_num, None)
    dataset = load_dataset(dataset_path, 2, function_from_touchstone, force_regen=False, random_sample=False)
    size = len(dataset)
    port_combinations = [(0, 0)]

    freqs = np.arange(0.1e9, 20e9, 0.03e9)
    init_poles = basic_poles_initialize(freqs_range=(0.1e9, 20e9), n_poles_real=4, n_poles_complex=2)
    for i in range(size):
        data = dataset[i]
        geo_params = torch.tensor(data["geo_params"], dtype=torch.float32)
        s_params = data["s_params"].reshape(port_num, port_num, -1)[0, 0].reshape(-1)

        poles, residues, d, h = vector_fit(freqs, s_params, init_poles, tol=1e-8, atol=1e-8, rtol=1e-8)
        
        # save the poles and residues
        filename = f"poles_residues_{i}.pkl"
        pr_data = {
            "poles": poles,
            "residues": residues,
            "d": d,
            "h": h,
        }
        file_path = os.path.join(dataset_path, filename)
        with open(file_path, "wb") as f:
            pickle.dump(pr_data, f)
        
        # if i < 10:
        #     s = frequency_to_s(freqs)
        #     s_params_fit = evaluate(poles, residues, d, h, s)
        #     # plot the poles and residues
        #     fig, axes = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)
        #     axes[0].plot(freqs, np.real(s_params), label='True Response', marker='o')
        #     axes[0].plot(freqs, np.real(s_params_fit), label='Fitted Response', linestyle='dashed')
        #     axes[1].plot(freqs, np.imag(s_params), label='True Response', marker='o')
        #     axes[1].plot(freqs, np.imag(s_params_fit), label='Fitted Response', linestyle='dashed')
        #     axes[2].plot(freqs, 20 * np.log10(np.abs(s_params)), label='True Response', marker='o')
        #     axes[2].plot(freqs, 20 * np.log10(np.abs(s_params_fit)), label='Fitted Response', linestyle='dashed')
        #     for ax in axes:
        #         ax.set_xlabel('Frequency (Hz)')
        #         axes[0].set_ylabel('Real')
        #         axes[1].set_ylabel('Imag')
        #         axes[2].set_ylabel("Magnitude (dB)")

        #     plt.legend()
        #     plt.savefig(f"pr_{i}.png")

if __name__ == "__main__":
    # test_pr()
    test_PolesResiduesDataset()