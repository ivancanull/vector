from ast import Lambda
from matplotlib import scale, tight_layout
from ..poles_init import basic_poles_initialize
from ..vector_fitting import vector_fit, evaluate, print_params, get_cindex
from ..error_estimate import fitted_error_estimate
from ..utils import frequency_to_s, add_frequency_point_on_have, map_to_available_freq
from ..complex_func import ComplexFunction, FunctionFromTouchstone
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t
from ..complex_func import FunctionAbstract
from ..vector_fitting import sort_poles_residues, construct_AB_matrices, calculate_zeros
import pymc as pm

DEBUG = True

def sample_r_d(freq, 
               Z, 
               init_poles, 
               max_iter=100, 
               tol=1e-8, 
               atol=1e-12,
               rtol=1e-12,
               d_fit=True, 
               h_fit=True,
               auto_scale=True,
               only_poles=False):
    """
    Fit a rational function to the given data using the Vector Fitting algorithm.
    
    Parameters:
    - freq: An array of frequency values (Hz).
    - Z: An array of complex f(s) values at the frequency points | ComplexFunction instance which represemts f(s).
    - init_poles: Initial poles provided by the user.
    - max_iter: The maximum number of iterations for the fitting process.
    - tol: The convergence tolerance.
    - relaxed: Whether to use d residues in Gustavsen's relocating method.
    - d_fit: Whether to fit the d parameter.
    - h_fit: Whether to fit the e parameter.
    - auto_scale: Whether to automatically scale the poles and residues.
    - only_poles: Whether to only fit the poles.
    
    Returns:
    - The fitted poles, residues, d, and h values.
    """

    d = 0
    h = 0
    N_poles = len(init_poles)
    poles = init_poles.copy()
    s = frequency_to_s(freq)
    Ns = len(s)

    # if Z is a given callable ComplexFunction instance, evaluate it at the frequency points
    if isinstance(Z, FunctionAbstract):
        Z = Z(s)
    else:
        Z = Z

    if auto_scale:
        # scale the s-domain by the maximum frequency
        s_scale = np.max(freq) / (2 * np.pi)
        s = s / s_scale
        poles = poles / s_scale

    poles = sort_poles_residues(poles, atol=atol, rtol=rtol)
    cindex = get_cindex(poles, atol=atol, rtol=rtol)

    # stage 1: poles and zeros identification
    A, B = construct_AB_matrices(s, cindex, poles, Z)
    cindex = get_cindex(poles, atol=atol, rtol=rtol)
    x = np.linalg.lstsq(A, B, rcond=None)[0]
    print("x: ", x)
    N_poles = len(poles)
    poles = calculate_zeros(poles, cindex, x[N_poles+2:])
    print("poles: ", poles)
    return A, B, x

def SimonBayesianAFS(Z, fmin: float, fmax: float, threshold: float = 0.1, sample_size: int = 500):
    """
    A Bayesian Approach to Adaptive Frequency Sampling implemented in Simon De Ridder et al.'s paper.

    Reference: De Ridder, Simon, et al. "A Bayesian approach to adaptive frequency sampling." 2019 IEEE 23rd Workshop on Signal and Power Integrity (SPI). IEEE, 2019.

    Args:
    - Z: a generator of complex data
    - fmin, fmax: range of frequency 
    - threshold: threshold for the weighted uncertainty
    - sample_size: number of samples from the posterior distribution
    """

    nF, respnF = 40, 100
    freq = np.linspace(fmin, fmax, nF)    

    # if Z is a FunctionFromTouchstone, map the frequency points to the available frequency points
    if isinstance(Z, FunctionFromTouchstone):
        respfreq = Z.freq
        respnF = len(respfreq)
    # if Z is a ComplexFunction, the available frequency points are the same as the response frequency points
    elif isinstance(Z, ComplexFunction):
        respfreq = np.linspace(fmin, fmax, respnF)
    else:
        raise ValueError("Z must be a FunctionFromTouchstone or ComplexFunction")

    freq = map_to_available_freq(freq, respfreq)
    if nF != len(freq):
        raise ValueError("Available frequency points are not enough")
    
    resps = frequency_to_s(respfreq)
    Zs = Z(resps)
    poles_num_max = 20

    if DEBUG:
        _, axes = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)
        axes[0].plot(respfreq, np.real(Zs), label='True Response', marker='o')
        axes[1].plot(respfreq, np.imag(Zs), label='True Response', marker='o')
        axes[2].plot(respfreq, 20 * np.log10(np.abs(Zs)), label='True Response', marker='o') 

    for order in range(min(nF + 1, poles_num_max - 1), min(nF + 1, poles_num_max)):
        s = frequency_to_s(freq)
        Zs = Z(s)

        # Unfortunately, it is not analytically possible to propagate (5) to the poles, 
        # as their calculation involves an eigenvalue problem.
        # Therefore, a set of poles is computed first.

        init_poles = basic_poles_initialize((fmin, fmax), 0 if order % 2 == 0 else 1, order // 2, scale="linear")
        assert len(init_poles) == order
        A, B, x = sample_r_d(freq, Z, init_poles, max_iter=10, tol=1e-8, atol=1e-8, only_poles=True)
    
        if DEBUG:
            print("x: ", x)
            print("A: ", A)
            print("B: ", B)

        B = B[:, None]

        n, p = A.shape # A: (n, p), x: (p, 1), B: (n, 1)
        print("n: ", n)
        print("p: ", p)
        sigma = 0.01

        with pm.Model() as model:
            # 先验分布 (假设 x 的先验是均值为 0，协方差为 I 的高斯分布)
            mu_0 = np.zeros(p)  # 先验均值
            Sigma_0 = np.eye(p)  # 先验协方差矩阵
            x = pm.MvNormal("x", mu=mu_0, cov=Sigma_0, shape=(p, 1))  # x 的先验分布

            # 似然函数
            b_obs = pm.Normal("b_obs", mu=pm.math.dot(A, x), sigma=sigma, observed=B)

            # 采样
            trace = pm.sample(500, tune=1000)
        
        pm.summary(trace)
        return
        x0 = np.zeros((p, 1))
        Lambda0 = np.eye(p) * 0
        alpha0 = 0.1
        beta0 = 0.1

        # calculate the posterior distribution
        Lambda_f = Lambda0 + A.T @ A # (p, p)
        print("Lambda_f shape: ", Lambda_f.shape)
        print("A shape: ", A.shape)
        print("B shape: ", B.shape)
        print("Lambda0 shape: ", Lambda0.shape)
        
        x_f = np.linalg.inv(Lambda_f) @ (Lambda0 @ x0 + A.T @ B) # (p, 1)
        print("x_f shape: ", x_f.shape)

        # compare x_f and x
        error = np.mean(x_f - x)
        print("MSE: ", error)

        alpha_f = alpha0 + n / 2
        beta_f = beta0 + 0.5 * (B.T @ B - x_f.T @ Lambda_f @ x_f + x0.T @ Lambda0 @ x0)
        print("alpha_f: ", alpha_f)
        print("beta_f: ", beta_f)

        # calculate the covariance matrix
        df = 2 * alpha_f
        scale = beta_f[0, 0] / alpha_f * np.linalg.inv(Lambda_f)

        print("scale shape: ", scale.shape)
        
        dist = multivariate_t(df=df, loc=x_f.flatten(), shape=scale)
        x_samples = dist.rvs(size=sample_size)
        
        for i in range(sample_size):
            error = x_samples[i] - x
            mse = np.mean(error ** 2)
            rmse = np.sqrt(mse) / np.mean(np.abs(x))
            print("MSE: ", mse)
            print("RMSE: ", rmse)
        
        c_res = x[:-1]
        d_res = x[-1]

        # check if d_res is suited for zeros calculation
        if np.abs(d_res) < tol:
            # d_res is too small, discard solution and proceed the |d_res| = tol_res
            print(f'Replacing d_res solution as it was too small ({d_res}).')
            d_res = tol if d_res > 0 else -tol

        # build test matrix H
        H = np.zeros((N_poles, N_poles), dtype=complex)
        for i, (ci, p) in enumerate(zip(cindex, poles)):
            if ci == 0:
                H[i, i] = p
                H[i] -= c_res / d_res
            elif ci == 1:
                x_, y = np.real(p), np.imag(p)
                H[i, i] = H[i + 1, i + 1] = x_
                H[i, i + 1] = -y
                H[i + 1, i] = y
                H[i] -= 2 * c_res / d_res

        poles = np.linalg.eigvals(H)
        
        unstable = np.real(poles) > 0
        poles[unstable] -= 2 * np.real(poles)[unstable]
        
        poles = sort_poles_residues(poles)
        
        # print samples
        for i in range(sample_size):
            reconstruct_residues = np.zeros_like(poles_fit, dtype=complex)
            for j, ci in enumerate(cindex):
                if ci == 0:
                    reconstruct_residues[i, j] = x_samples[i, j]
                elif ci == 1:
                    reconstruct_residues[i, j] = x_samples[i, j] + x_samples[i, j + 1] * 1j 
                    reconstruct_residues[i, j + 1] = x_samples[i, j] - x_samples[i, j + 1] * 1j 

            print("d sample: ", x_samples[i, -2])
            print("h sample: ", x_samples[i, -1])
            Z_fit = A @ x_samples[i]
            axes[0].plot(freq, np.real(Z_fit), linestyle='dashed', linewidth=1, alpha=0.5)
            axes[1].plot(freq, np.imag(Z_fit), linestyle='dashed', linewidth=1, alpha=0.5)
            axes[2].plot(freq, 20 * np.log10(np.abs(Z_fit)), linestyle='dashed', linewidth=1, alpha=0.5)

    for ax in axes:
        ax.set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Real')
    axes[1].set_ylabel('Imag')
    axes[2].set_ylabel("Magnitude (dB)")
    plt.legend()
    plt.savefig('test_lbvf.png')

    return


                
        