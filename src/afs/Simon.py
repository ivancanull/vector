from turtle import up
from ..poles_init import basic_poles_initialize
from ..vector_fitting import vector_fit, evaluate, print_params, get_cindex
from ..error_estimate import fitted_error_estimate
from ..utils import frequency_to_s, add_frequency_point_on_have, map_to_available_freq
from ..complex_func import ComplexFunction, FunctionFromTouchstone
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t, multivariate_normal, invgamma
from ..complex_func import FunctionAbstract
from ..vector_fitting import sort_poles_residues, construct_AB_matrices, calculate_zeros
import pymc as pm

DEBUG = True

def sample_r_d(A, B, x, sample_size):
    if B.ndim == 1:
        B = B[:, None]
        nB = B.shape[0]
    n, p = A.shape # A: (n, p), x: (p, 1), B: (n, 1)
    x0 = np.zeros((p, 1))
    Lambda0 = np.eye(p) * 1e-6
    alpha0 = 0
    beta0 = 0

    # calculate the posterior distribution
    Lambda_f = Lambda0 + A.T @ A # (p, p)
    Lambda_f_inv = np.linalg.inv(Lambda_f)
    # check if Lambda_f is symmetric positive-semidefinite
    eigenvalues = np.linalg.eigvals(Lambda_f_inv)
    if np.all(eigenvalues > 0):
        print("Lambda_f is positive definite.")
    else:
        print("Lambda_f is NOT positive definite.")
        Lambda_f_inv = 0.5 * (Lambda_f_inv + Lambda_f_inv.T) 

    x_f = Lambda_f_inv @ (Lambda0 @ x0 + A.T @ B) # (p, 1)
    alpha_f = alpha0 + nB / 2
    beta_f = beta0 + 0.5 * (B.T @ B - x_f.T @ Lambda_f @ x_f + x0.T @ Lambda0 @ x0)

    sigma_sq_samples = invgamma.rvs(a=alpha_f, scale=beta_f, size=sample_size)
    print("sigma_sq_samples: ", sigma_sq_samples)

    x_samples = []
    for sigma_sq in sigma_sq_samples:
        cov_matrix = sigma_sq * Lambda_f_inv
        # # cov_matrix_sym_pos_semidefi = 0.5 * (cov_matrix + cov_matrix.T)
        # cov_matrix_sym_pos_semidefi = cov_matrix + 1e-8 * np.eye(cov_matrix.shape[0])
        x_sample = multivariate_normal.rvs(mean=x_f.flatten(), cov=cov_matrix)
        x_samples.append(x_sample)

    return np.array(x_samples)


    # calculate the covariance matrix
    df = 2 * alpha_f
    scale_inv = alpha_f / beta_f[0, 0] * Lambda_f
    print("scale_inv: ", scale_inv)
    scale = np.linalg.inv(alpha_f / beta_f[0, 0] * Lambda_f)
    
    dist = multivariate_t(df=df, loc=x_f.flatten(), shape=scale)
    x_samples = dist.rvs(size=sample_size)
    return x_samples
        
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

    order = 6
    init_poles = basic_poles_initialize((fmin, fmax), 0 if order % 2 == 0 else 1, order // 2, scale="linear")
    A, B, x = vector_fit(freq, Z, init_poles, max_iter=1, tol=1e-8, atol=1e-6, method="SimonAFS")

    print("x: ", x)
    print("A: ", A)
    print("B: ", B)

    x_samples = sample_r_d(A, B, x, sample_size=100)
    print(x_samples.shape)

    # plot error bars
    if DEBUG:
        fig, axes = plt.subplots(figsize=(8, 6))
        x_ax = np.arange(1, len(x) + 1)
        # 
        parts = axes.violinplot(x_samples,
                        showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        axes.scatter(x_ax, x, marker='o',)
        axes.set_xlabel('Pole index')
        axes.set_ylabel('Residue')
        # axes.set_ylim(bottom=min(np.min(x), np.min(x_samples)), top=max(np.max(x), np.max(x_samples)))
        axes.set_title('Residues with error bars')
        plt.savefig('test_lbvf_error_bars.png')

    return

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


                
        