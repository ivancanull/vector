from src import *

import numpy as np
import matplotlib.pyplot as plt

# Example usage
poles = [
    -4500,
    -41000,
    -100+5000j, -100-5000j,
    -120+15000j, -120-15000j,
    -3000+35000j, -3000-35000j,
    -200+45000j, -200-45000j,
    -1500+45000j, -1500-45000j,
    -500+70000j, -500-70000j,
    -1000+73000j, -1000-73000j,
    -2000+90000j, -2000-90000j,
]
residues = [
    -3000,
    -83000,
    -5+7000j, -5-7000j,
    -20+18000j, -20-18000j,
    6000+45000j, 6000-45000j,
    40+60000j, 40-60000j,
    90+10000j, 90-10000j,
    50000+80000j, 50000-80000j,
    1000+45000j, 1000-45000j,
    -5000+92000j, -5000-92000j
]
d = .2
h = 2e-5

class ScaleFunction(ComplexFunction):
    def __init__(self, poles, residues, d=0, h=0, s_min=0, s_scale=1):
        
        if len(poles) != len(residues):
            raise ValueError("Poles and residues must be of the same length.")
        
        self.poles = np.array(poles)
        self.residues = np.array(residues)
        self.d = d
        self.h = h
        self.s_min = s_min
        self.s_scale = s_scale

    def evaluate(self, s):
        s = (s - self.s_min) / self.s_scale
        s = np.array(s, dtype=complex)  # Ensure s is a numpy array
        value = self.d + self.h * s  # Initialize with d + hs (handling the polynomial term)

        # Compute the sum of residues/(s - poles)
        for pole, residue in zip(self.poles, self.residues):
            value += residue / (s - pole)
        
        return value

def read_touchstone():
    touchstone = TorchstoneParser(2)
    data = touchstone.parse("data/coupled_line_filter_touchstone.s2p")
    Z = FunctionFromTouchstone("data/coupled_line_filter_touchstone.s2p", 2, [0, 0])
    return Z

def _standard_error_estimate(Z, Z_fit):
    # compare the true response with the fitted response
    for method in ["RMSE", "NRMSE", "R-squared", "MReE", "MxReE"]:
        if method == "NRMSE":
            error = fitted_error_estimate(Z, Z_fit, method=method, norm="range")
        else:
            error = fitted_error_estimate(Z, Z_fit, method=method)
        if method == "MReE" or method == "MxReE":
            print(f"{method}: {error * 100}%")
        else:
            print(f"{method}: {error}")

def test_abstract_function_vector_fitting(fmin, fmax, n_poles_real, n_poles_complex, max_iter, freq, response, test_name, method):

    s = frequency_to_s(freq)
    init_poles = basic_poles_initialize(freqs_range=(fmin, fmax), n_poles_real=n_poles_real, n_poles_complex=n_poles_complex)

    # start the fitting process
    start_time = time.perf_counter()
    poles_fit, poles_residues, poles_d, poles_h = vector_fit(freq, response, init_poles, max_iter=max_iter, tol=1e-6, atol=1e-6, method=method)
    print(f"Time taken: {time.perf_counter() - start_time:.2f}s")

    # evaluate the fitted response
    Z = response(s)
    Z_fit = evaluate(poles_fit, poles_residues, poles_d, poles_h, s)
    _standard_error_estimate(Z, Z_fit)

    # plot the results
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)
    axes[0].plot(freq, np.real(Z), label='True Response', marker='o')
    axes[0].plot(freq, np.real(Z_fit), label='Fitted Response', linestyle='dashed')
    axes[1].plot(freq, np.imag(Z), label='True Response', marker='o')
    axes[1].plot(freq, np.imag(Z_fit), label='Fitted Response', linestyle='dashed')
    axes[2].plot(freq, 20 * np.log10(np.abs(Z)), label='True Response', marker='o')
    axes[2].plot(freq, 20 * np.log10(np.abs(Z_fit)), label='Fitted Response', linestyle='dashed')
    for ax in axes:
        ax.set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Real')
    axes[1].set_ylabel('Imag')
    axes[2].set_ylabel("Magnitude (dB)")

    plt.legend()
    plt.savefig(f"{test_name}.pdf")
    plt.savefig(f"{test_name}.png")

def test_defined_function_vector_fitting(method="pole_relocating"):
    print(f"\nTesting {method} method with defined")
    # define the function and frequency sample points
    freq = np.linspace(0, 2e4, 1000)
    s = frequency_to_s(freq)
    response = ComplexFunction(poles, residues, d, h)

    test_abstract_function_vector_fitting(1, 2e4, 2, 8, 10, freq, response, f"test_defined_function_vector_fitting_{method}", method)

def test_data_function_vector_fitting(method="pole_relocating"):
    print(f"\nTesting {method} method with data")
    response = _read_touchstone()
    fmin, fmax = 3.0e9, 4.2e9
    freq = np.arange(3, 4.21, 0.01) * 1e9
    s = frequency_to_s(freq)
    test_abstract_function_vector_fitting(fmin, fmax, 0, 20, 20, freq, response, f"test_data_function_vector_fitting_{method}", method)

def test_vector_fitting():
    test_defined_function_vector_fitting("pole_relocating")
    test_data_function_vector_fitting("pole_relocating")  
    test_defined_function_vector_fitting("Gustavsen")
    test_data_function_vector_fitting("Gustavsen")    
        

def test_complex_reshape():
    db = -29.65140962642262
    degree = 55.75907076987683

    A = db * np.exp(1j * np.radians(degree))
    print(np.abs(A), np.angle(A))
    A_trans_abs, A_trans_angle = np.abs(A), np.angle(A)
    actual_A_2 = 10 ** (A_trans_abs / 20) * np.exp(1j * A_trans_angle)
    actual_A = 10 ** (db / 20) * np.exp(1j * np.radians(degree))
    print(actual_A)
    print(actual_A_2)
    print("Difference: ", np.abs(actual_A - actual_A_2))
    
def test_Giulio_afs():
    Z = _read_touchstone()
    GiulioAFS(Z, 3e9, 4.2e9, n_min=2, n_max=10, tol_error=1e-6, heuristics_tol_error=[1e-4, 1e-4, 1e-4, 1e-4])


def test_scale():
    # Define the function and frequency sample points
    origin_freq = np.linspace(0, 2e4, 1000)

    s_scale = 1.2e9 / 2e4
    s_min = frequency_to_s(3.0e9)

    response = ScaleFunction(poles, residues, d, h, s_min, s_scale)
    freq = np.linspace(3.0e9, 4.2e9, 100)
    s = frequency_to_s(freq)
    np.set_printoptions(precision=3)
    init_poles = basic_poles_initialize(freqs_range=(3.0e9, 4.2e9), n_poles_real=2, n_poles_complex=8)

    poles_fit, poles_residues, poles_d, poles_h = vector_fit(freq, response, init_poles, max_iter=10, tol=1e-6, relaxed=True)
    Z = response(s)
    Z_fit = evaluate(poles_fit, poles_residues, poles_d, poles_h, s)


    fig, axes = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)
    axes[0].plot(freq, np.real(Z), label='True Response', marker='o')
    axes[0].plot(freq, np.real(Z_fit), label='Fitted Response', linestyle='dashed')
    axes[1].plot(freq, np.imag(Z), label='True Response', marker='o')
    axes[1].plot(freq, np.imag(Z_fit), label='Fitted Response', linestyle='dashed')
    axes[2].plot(freq, 20 * np.log10(np.abs(Z)), label='True Response', marker='o')
    axes[2].plot(freq, 20 * np.log10(np.abs(Z_fit)), label='Fitted Response', linestyle='dashed')
    for ax in axes:
        ax.set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Real')
    axes[1].set_ylabel('Imag')
    axes[2].set_ylabel("Magnitude (dB)")


    plt.legend()
    plt.savefig("test_vf_scale.pdf")
    plt.savefig("test_vf_scale.png")

def test_lbvf():
    # Z = read_touchstone()
    freq = np.linspace(0, 2e4, 1000)
    s = frequency_to_s(freq)
    response = ComplexFunction(poles, residues, d, h)
    SimonBayesianAFS(Z=response, fmin=1, fmax=2e4, sample_size=10)

if __name__ == "__main__":
    # test_SPDataset()
    # test_touchstone()
    # test_complex_reshape()    
    # test_afs()
    # test_lbvf()
    # test_vf()
    # test_load_dataset()
    test_SPDataset()
    test_multi_output_bnn()