from src.touchstone import *
from src.complex_func import *
from src.utils import *
from src.poles_init import *
from src.afs import *
from src.vector_fitting import *

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
    
def test_init_poles():
    init_poles = basic_poles_initialize([3e9, 4.2e9], 0, 2)
    print(init_poles)

def test_touchstone():
    Z = read_touchstone()
    fmin = 3.0e9 
    fmax = 4.2e9
    freq = np.arange(3, 4.21, 0.01) * 1e9
    s = frequency_to_s(freq)
    values = Z(s)
    S11 = values
    n_min, n_max = 6, 14

    respnF = 100
    respfreq = np.linspace(fmin, fmax, respnF)
    available_freq = Z.freq
    resps = frequency_to_s(map_to_available_freq(respfreq, available_freq))
    respnF = len(respfreq)
    resps = frequency_to_s(respfreq)
    best_ser = {}
    second_best_ser = {}
    for order in range(n_min, n_max + 1):
        init_poles = basic_poles_initialize((fmin, fmax), 0 if order % 2 == 0 else 1, order // 2, scale="linear")
        assert len(init_poles) == order
        poles_fit, poles_residues, poles_d, poles_h = vector_fit(freq, Z, init_poles, max_iter=50, tol=1e-8, atol=1e-6)
        Z_fit = evaluate(poles_fit, poles_residues, poles_d, poles_h, s)                
        error = fitted_error_estimate(Z(s), Z_fit)

        # save and update best and second best series
        if best_ser == {}:
            best_ser = {"poles": poles_fit, "residues": poles_residues, "d": poles_d, "h": poles_h, "error": error}
        elif second_best_ser == {}:
            if error < best_ser["error"]:
                second_best_ser = best_ser
                best_ser = {"poles": poles_fit, "residues": poles_residues, "d": poles_d, "h": poles_h, "error": error}
            else:
                second_best_ser = {"poles": poles_fit, "residues": poles_residues, "d": poles_d, "h": poles_h, "error": error}
        else:
                    if error < best_ser["error"]:
                        second_best_ser = best_ser
                        best_ser = {"poles": poles_fit, "residues": poles_residues, "d": poles_d, "h": poles_h, "error": error}
                    elif error < second_best_ser["error"]:
                        second_best_ser = {"poles": poles_fit, "residues": poles_residues, "d": poles_d, "h": poles_h, "error": error}

    # adaptive sampling loop
    resp1 = evaluate(best_ser["poles"], best_ser["residues"], best_ser["d"], best_ser["h"], resps)
    resp2 = evaluate(second_best_ser["poles"], second_best_ser["residues"], second_best_ser["d"], second_best_ser["h"], resps)
    print("poles num best: ", len(best_ser["poles"]))
    print("poles num second best: ", len(second_best_ser["poles"]))

    h0 = np.max(np.abs(resp1 - resp2) / np.abs(resp1)) # relative error
    h1 = np.max((np.abs(resp1) - np.abs(resp2)) / np.abs(resp1)) # magnitude error
    h2 = np.max(np.abs(np.unwrap(np.angle(resp1) - np.angle(resp2)))) # phase error
    h3 = np.mean(np.abs(resp1 - resp2) ** 2) # mean square error of the two responses
    h = [h0, h1, h2, h3]          
    for i, hi in enumerate(h):
        print(f"h{i}: {hi}")

    """
    h0: 2.6550193661069414e-05
    h1: 1.3744714174066998e-05
    h2: 2.6458499113868328e-05
    h3: 1.3275812155468643e-13
    """

    # init_poles = basic_poles_initialize(freqs_range=(3.0e9, 4.2e9), n_poles_real=0, n_poles_complex=1)

    # poles_fit, poles_residues, poles_d, poles_h = vector_fit(freq, Z, init_poles, max_iter=50, tol=1e-6, relaxed=True)
    # Z_fit = evaluate(poles_fit, poles_residues, poles_d, poles_h, s)          

    # fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    # ax.plot(freq, 20 * np.log10(np.abs(S11)), label='True Response', marker='o')
    # ax.plot(freq, 20 * np.log10(np.abs(Z_fit)), label='Fitted Response', linestyle='dashed')
    # ax.set_xlabel('Frequency (Hz)')

    # ax.set_ylabel("Magnitude (dB)")

    # plt.legend()
    # plt.savefig("test_touchstone.png")

def test_afs():
    Z = read_touchstone()
    GiulioAFS(Z, 3e9, 4.2e9, n_min=2, n_max=10, tol_error=1e-6, heuristics_tol_error=[1e-4, 1e-4, 1e-4, 1e-4])

def test_vf():
    freq = np.linspace(0, 2e4, 1000)
    s = frequency_to_s(freq)
    response = ComplexFunction(poles, residues, d, h)
    init_poles = basic_poles_initialize(freqs_range=(1, 2e4), n_poles_real=2, n_poles_complex=8)
    poles_fit, poles_residues, poles_d, poles_h = vector_fit(freq, response, init_poles, max_iter=10, tol=1e-6)
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
    plt.savefig("test_vf.pdf")
    plt.savefig("test_vf.png")

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
    # test_touchstone()
    # test_complex_reshape()    
    # test_afs()
    test_lbvf()
    # test_vf()

