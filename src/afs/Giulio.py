try:
    from ..poles_init import basic_poles_initialize
    from ..vector_fitting import vector_fit, evaluate, print_params, get_cindex
    from ..error_estimate import fitted_error_estimate
    from ..utils import frequency_to_s, add_frequency_point_on_have, map_to_available_freq
    from ..complex_func import ComplexFunction, FunctionFromNumpy, FunctionFromTouchstone
except ImportError:  # pragma: no cover - PYTHONPATH=src import style
    from poles_init import basic_poles_initialize
    from vector_fitting import vector_fit, evaluate, print_params, get_cindex
    from error_estimate import fitted_error_estimate
    from utils import frequency_to_s, add_frequency_point_on_have, map_to_available_freq
    from complex_func import ComplexFunction, FunctionFromNumpy, FunctionFromTouchstone
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Callable

DEBUG = False

def GiulioAFS(Z, fmin: float, fmax: float, n_min: int = 1, n_max: int = 20, tol_error: float = 1e-4, heuristics_tol_error: Tuple[float, float, float, float] = (1e-4, 1e-4, 1e-4, 1e-4), max_n_freq = 20):
    """
    Adaptive Frequency Sampling Algorithm implemented in Giulio A. et al.'s paper.

    Reference: Antonini, Giulio, Dirk Deschrijver, and Tom Dhaene. "Broadband rational macromodeling based on the adaptive frequency sampling algorithm and the partial element equivalent circuit method." IEEE Transactions on Electromagnetic Compatibility 50.1 (2008): 128-137.

    Args:
    - Z: a generator of complex data
    - fmin, fmax: range of frequency 
    - n_min, n_max: minimum and maximum order of the model
    - tol_error: tolerance for the overall RMS error 
    - heuristics_tol_error: tuple of tolerances for the heuristics (h0, h1, h2, h3)
    """

    
    convergence = False
    nF, respnF = 4, 100

    freq = np.linspace(fmin, fmax, nF)    

    # if Z is a FunctionFromTouchstone, map the frequency points to the available frequency points
    if isinstance(Z, FunctionFromTouchstone):
        respfreq = Z.freq
        respnF = len(respfreq)
    # if Z is a ComplexFunction, the available frequency points are the same as the response frequency points
    elif isinstance(Z, ComplexFunction):
        respfreq = np.linspace(fmin, fmax, respnF)
    elif isinstance(Z, FunctionFromNumpy):
        respfreq = Z.freq
        respnF = len(respfreq)
    else:
        raise ValueError("Z must be a FunctionFromTouchstone or ComplexFunction")

    freq = map_to_available_freq(freq, respfreq)
    if nF != len(freq):
        raise ValueError("Available frequency points are not enough")
    
    resps = frequency_to_s(respfreq)

    while not convergence:

        # adaptive modeling loop
        order_found = False
        s = frequency_to_s(freq)
        Zs = Z(s)
        
        while not order_found:
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
                        
            if best_ser["error"] > tol_error:
                n_min += 1
                n_max += 1
                if n_max > 50:
                    raise ValueError("Model order too high")
            else:
                if DEBUG:
                    print("Model order found: ", len(best_ser["poles"]), "Error: ", best_ser["error"])
                order_found = True
        
        # adaptive sampling loop
        resp1 = evaluate(best_ser["poles"], best_ser["residues"], best_ser["d"], best_ser["h"], resps)
        resp2 = evaluate(second_best_ser["poles"], second_best_ser["residues"], second_best_ser["d"], second_best_ser["h"], resps)

        h0 = np.max(np.abs(resp1 - resp2) / np.abs(resp1)) # relative error
        h1 = np.max((np.abs(resp1) - np.abs(resp2)) / np.abs(resp1)) # magnitude error
        h2 = np.max(np.abs(np.unwrap(np.angle(resp1) - np.angle(resp2)))) # phase error
        h3 = np.mean(np.abs(resp1 - resp2) ** 2) # mean square error of the two responses
        h = [h0, h1, h2, h3]
        
        not_satisfied = []
        for i,  hi in enumerate(h):
            if hi > heuristics_tol_error[i]:
                not_satisfied.append(f"h{i}")

        if DEBUG:
            if not_satisfied == []:
                print("Satisfied")
            else:
                print("Not satisfied: ", {hi: h[i] for i, hi in enumerate(not_satisfied)})    
            
        if not_satisfied == [] or nF == max_n_freq:
            convergence = True
        else:
            if DEBUG and nF % 10 == 0:
                # plot the responses
                _, axes = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)
                axes[0].plot(freq, np.real(Zs), label='True Response', marker='o')
                axes[1].plot(freq, np.imag(Zs), label='True Response', marker='o')
                axes[2].plot(freq, 20 * np.log10(np.abs(Zs)), label='True Response', marker='o') 
                axes[0].plot(respfreq, np.real(resp1), label='Fitted Response', linestyle='dashed')
                axes[1].plot(respfreq, np.imag(resp1), label='Fitted Response', linestyle='dashed')
                axes[2].plot(respfreq, 20 * np.log10(np.abs(resp1)), label='Fitted Response', linestyle='dashed')
                for ax in axes:
                    ax.set_xlabel('Frequency (Hz)')
                axes[0].set_ylabel('Real')
                axes[1].set_ylabel('Imag')
                axes[2].set_ylabel("Magnitude (dB)")
                plt.legend()
                plt.savefig(f"test_afs/test_afs_nF_{nF}.png")
                plt.cla()

            # get the frequency point with the most |resp1 - resp2| / resp1| while unsampled
            idx = np.argsort(np.abs(resp1 - resp2) / np.abs(resp1))[::-1]
            added = False
            while not added and len(idx) > 0:
                new_freq = respfreq[idx[0]]
                added, freq = add_frequency_point_on_have(freq, respfreq, new_freq)
                if not added:
                    idx = idx[1:]
                    # if DEBUG:
                    #     print("Frequency point already sampled: %.2e GHz" % (new_freq / 1e9))
                else:
                    if DEBUG:
                        print("New frequency point added: %.2e GHz" % (new_freq / 1e9))
            nF = len(freq)
            if DEBUG:
                print("Number of frequency points: ", nF)
    
    # final test
    if DEBUG:
        

        Z_fit = evaluate(best_ser["poles"], best_ser["residues"], best_ser["d"], best_ser["h"], resps)
        Zs = Z(resps)
        error = fitted_error_estimate(Z=Zs, Z_fit=Z_fit)
        print_params(cindex=get_cindex(best_ser["poles"]), poles=best_ser["poles"], residues=best_ser["residues"], d=best_ser["d"], h=best_ser["h"])
        print("Final RMSE: ", error)

        _, axes = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)
        axes[0].plot(respfreq, np.real(Zs), label='True Response', marker='o')
        axes[1].plot(respfreq, np.imag(Zs), label='True Response', marker='o')
        axes[2].plot(respfreq, 20 * np.log10(np.abs(Zs)), label='True Response', marker='o') 
        axes[0].plot(respfreq, np.real(Z_fit), label='Fitted Response', linestyle='dashed')
        axes[1].plot(respfreq, np.imag(Z_fit), label='Fitted Response', linestyle='dashed')
        axes[2].plot(respfreq, 20 * np.log10(np.abs(Z_fit)), label='Fitted Response', linestyle='dashed')
        for ax in axes:
            ax.set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Real')
        axes[1].set_ylabel('Imag')
        axes[2].set_ylabel("Magnitude (dB)")
        plt.legend()
        plt.savefig(f"test_afs/test_afs_nF_final_{nF}.png")
        plt.cla()
    
    return best_ser, second_best_ser
            
    
                    
            

                    
            
            




            
