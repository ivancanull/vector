from re import DEBUG
from .poles_init import basic_poles_initialize
from .vector_fitting import vector_fit, evaluate
from .error_estimate import fitted_error_estimate
from .utils import frequency_to_s
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Callable

DEBUG = True
def GiulioAFS(Z, fmin: float, fmax: float, n_min: int = 1, n_max: int = 20, tol_order: float = 1e-6):
    """
    Adaptive Frequency Sampling Algorithm implemented in Giulio A. et al.'s paper.

    Reference: Antonini, Giulio, Dirk Deschrijver, and Tom Dhaene. "Broadband rational macromodeling based on the adaptive frequency sampling algorithm and the partial element equivalent circuit method." IEEE Transactions on Electromagnetic Compatibility 50.1 (2008): 128-137.

    Args:
    - Z: a generator of complex data
    - fmin, fmax: range of frequency 
    - poles_initializer: methods to initiate poles
    """

    
    convergence = False
    nF = 10
    freq = np.linspace(fmin, fmax, nF)
    s = frequency_to_s(freq)
    while not convergence:
        # adaptive modeling loop
        order_found = False
        order = n_min
        Zs = Z(s)
        if DEBUG:
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)
            axes[0].plot(freq, np.real(Zs), label='True Response', marker='o')
            axes[1].plot(freq, np.imag(Zs), label='True Response', marker='o')
            axes[2].plot(freq, 20 * np.log10(np.abs(Zs)), label='True Response', marker='o')
        while not order_found:
            init_poles = basic_poles_initialize((fmin, fmax), 0 if order % 2 == 0 else 1, order // 2, scale="linear")
            assert len(init_poles) == order
            poles_fit, poles_residues, poles_d, poles_h = vector_fit(freq, Z, init_poles, max_iter=10, tol=1e-6, relaxed=True)
            Z_fit = evaluate(poles_fit, poles_residues, poles_d, poles_h, s)

            error = fitted_error_estimate(Z(s), Z_fit)
            if DEBUG:
                axes[0].plot(freq, np.real(Z_fit), label='Fitted Response', linestyle='dashed')
                axes[1].plot(freq, np.imag(Z_fit), label='Fitted Response', linestyle='dashed')
                axes[2].plot(freq, 20 * np.log10(np.abs(Z_fit)), label='Fitted Response', linestyle='dashed')
            if order == n_min:
                prev_error = error
                order += 1
                print("Initial error: ", error)
            elif np.abs(error - prev_error) < tol_order or order == n_max :
                order_found = True
            else:
                
                if DEBUG:
                    print("Order: ", order, "Error: ", error, "Prev Error: ", prev_error, "Diff: ", np.abs(error - prev_error))
                prev_error = error
                order += 1
        if DEBUG:
            for ax in axes:
                ax.set_xlabel('Frequency (Hz)')
            axes[0].set_ylabel('Real')
            axes[1].set_ylabel('Imag')
            axes[2].set_ylabel("Magnitude (dB)")
            plt.legend()
            plt.savefig("test_afs.png")
        convergence = True
                    
            

                    
            
            




            
