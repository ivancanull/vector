from typing import Tuple, List
import numpy as np
from .utils import frequency_to_s, get_cindex, sort_poles_residues
from .error_estimate import fitted_error_estimate
from .complex_func import FunctionAbstract

VF_DEBUG = False
__all__ = ["vector_fit", "evaluate"]

def print_params(cindex, poles, residues=None, d=0, h=0, postfix=""):
    """
    Print the poles, residues, d, and h values. 
    """
    # cfmt = lambda x: f'{x.real:.2E} {"+-"[x.imag > 0]} {abs(x.imag):.2E}j'
    if residues is not None:
        print(f'Poles {postfix}\t\t\t\tResidues {postfix}')
    else:
        print(f'Poles {postfix}')
    for i, (p, r) in enumerate(zip(poles, residues)):
        if cindex[i] == 0:
            print(f'{p.real:.2E}\t\t\t{r.real:.2E}')
        elif cindex[i] == 1:
            print(f'{p.real:.2E} +- {abs(p.imag):.2E}j\t\t{r.real:.2E} +- {abs(r.imag):.2E}j\t')
        else:
            pass
    print ("d " + postfix + ": " + "{:.2E}".format(d))
    print ("h " + postfix + ": " + "{:.2E}".format(h))

def evaluate(poles, residues, d, h, s):
    """
    Evaluate the rational function at given complex frequency points s.
    
    Parameters:
    - s: A complex number or array of complex numbers where the function is to be evaluated.
    
    Returns:
    - The value of the rational function at s or an array of values if s is an array.
    """
    f = d + h * s

    for i in range(len(poles)):
        f += residues[i] / (s - poles[i])
    return f

def construct_AB_matrices(s: np.ndarray,
                          Z: np.ndarray,
                          d_fit: bool,
                          h_fit: bool,
                          Ns: int,
                          N_poles: int,
                          poles: np.ndarray,
                          cindex: List[int]):
    """
    Construct the A and B matrices for the linear system Ax = B, ensuring that poles and residues are conjugate pairs.
    """
    # stage 1: poles and zeros identification
    A = np.zeros((Ns, N_poles * 2 + 2), dtype=complex)
    # [c1, c2, ..., cn, d, e, c1_res, c2_res, ..., cn_res]
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1 / (s - p)
        elif cindex[i] == 1:
            A[:, i] = 1 / (s - p) + 1 / (s - np.conj(p))
        elif cindex[i] == 2:
            A[:, i] = 1j / (s - p) - 1j / (s - np.conj(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))
        A[:, N_poles + 2 + i] = -A[:, i] * Z

    if d_fit:
        A[:, N_poles] = 1
    if h_fit:
        A[:, N_poles + 1] = s
    # A[:, N_poles * 2 + 2] = -1 * Z

    A = np.vstack([A.real, A.imag])
    B = np.hstack([Z.real, Z.imag])

    Q, R = np.linalg.qr(A)

    if R.shape[0] < R.shape[1]:
        m = R.shape[0]
        R = R[:m, :m]
        Q = Q[:, :m]
        
    Qt_B = np.dot(Q.T, B)
    x = np.linalg.solve(R, Qt_B)

    return R, Qt_B, x

def calculate_zeros(poles, cindex, C):
    """
    Calculate the zeros of the rational function.
    """
    N_poles = len(poles)
    A = np.diag(poles)
    B = np.ones(N_poles)
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = np.real(p), np.imag(p)
            A[i, i] = A[i + 1, i + 1] = x
            A[i, i + 1] = -y
            A[i + 1, i] = y
            B[i] = 2
            B[i + 1] = 0
    
    H = np.real(A - np.outer(B, C))
    poles = np.linalg.eigvals(H)
    unstable = np.real(poles) > 0
    poles[unstable] -= 2 * np.real(poles)[unstable]
    return poles

def _Gustavsen_vector_fit(s: np.ndarray,
                          Z: np.ndarray,
                          d_fit: bool,
                          h_fit: bool,
                          Ns: int,
                          N_poles: int,
                          poles: np.ndarray,
                          cindex: List[int],
                          ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Fit a rational function to the given data using the Vector Fitting algorithm, based on Gustavsen's relocating method.

    Matrix A_bar is defined as:
    A_bar = [
        [1/(S1 - a1_0), ..., 1/(S1 - aN_0), 1, S1, -S1/(S1 - a1_0), ..., -S1/(S1 - aN_0)],
        [⋮, ⋱, ⋮, ⋮, ⋮, ⋮, ⋮, ⋮],
        [1/(S_N5 - a1_0), ..., 1/(S_N5 - aN_0), 1, S_N5, -S_N5/(S_N5 - a1_0), ..., -S_N5/(S_N5 - aN_0)]
    ]

    Vector x_bar is defined as:
    x_bar = [r1, ..., rN, d, e, r_tilde1, ..., r_tildeN]^T

    Vector b_bar is defined as:
    b_bar = [S1, ..., S_N5]^T
    
    """

    # stage 1: poles and zeros identification
    A = np.zeros((Ns, N_poles * 2 + 2), dtype=complex)
    # [c1, c2, ..., cn, d, e, c1_res, c2_res, ..., cn_res]
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1 / (s - p)
        elif cindex[i] == 1:
            A[:, i] = 1 / (s - p) + 1 / (s - np.conj(p))
        elif cindex[i] == 2:
            A[:, i] = 1j / (s - p) - 1j / (s - np.conj(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))
        A[:, N_poles + 2 + i] = -A[:, i] * Z

    if d_fit:
        A[:, N_poles] = 1
    if h_fit:
        A[:, N_poles + 1] = s
    # A[:, N_poles * 2 + 2] = -1 * Z

    A = np.vstack([A.real, A.imag])
    B = np.hstack([Z.real, Z.imag])

    Q, R = np.linalg.qr(A)
    Qt_B = np.dot(Q.T, B)
    x = np.linalg.solve(R, Qt_B)

    A = np.diag(poles)
    B = np.ones(N_poles)
    C = x[N_poles + 2:]
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = np.real(p), np.imag(p)
            A[i, i] = A[i + 1, i + 1] = x
            A[i, i + 1] = -y
            A[i + 1, i] = y
            B[i] = 2
            B[i + 1] = 0
    
    H = np.real(A - np.outer(B, C))
    poles = np.linalg.eigvals(H)
    unstable = np.real(poles) > 0
    poles[unstable] -= 2 * np.real(poles)[unstable]
    return poles

def _pole_relocating_vector_fit(s: np.ndarray,
                                Z: np.ndarray,
                                d_fit: bool,
                                h_fit: bool,
                                Ns: int,
                                N_poles: int,
                                poles: np.ndarray,
                                cindex: List[int],
                                tol_res: float
                                ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    
    # stage 1: poles and zeros identification
    A = np.zeros((Ns, N_poles * 2 + 3), dtype=complex)
    # [c1, c2, ..., cn, d, e, c1_res, c2_res, ..., cn_res, d_res]
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1 / (s - p)
        elif cindex[i] == 1:
            A[:, i] = 1 / (s - p) + 1 / (s - np.conj(p))
        elif cindex[i] == 2:
            A[:, i] = 1j / (s - p) - 1j / (s - np.conj(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))
        A[:, N_poles + 2 + i] = -A[:, i] * Z

    if d_fit:
        A[:, N_poles] = 1
    if h_fit:
        A[:, N_poles + 1] = s
    A[:, N_poles * 2 + 2] = -1 * Z

    # sum up the real of the poles for future use
    trival_coeff = np.sum(A[:, :N_poles].real, axis=0)

    A = np.vstack([A.real, A.imag])
    R = np.linalg.qr(A, mode='r')
    dim_k = min(2 * Ns, N_poles * 2 + 3)
    if dim_k < N_poles * 2 + 3:
        r12_nrows = Ns
        r22_nrows = Ns
    else:
        r12_nrows = N_poles + 2
        r22_nrows = N_poles + 1
    r12_ncols = N_poles + 2
    r22_ncols = N_poles + 1

    # restrict the d_res to be non-zero
    R22 = R[r12_nrows:, r12_ncols:]
    
    R22 = np.vstack([R22, np.zeros((1, r22_ncols))])
    R22[-1, :-1] = np.real(trival_coeff)
    R22[-1, -1] = Ns

    # weight the last d_res LS problem
    weight = np.sqrt(np.linalg.norm(Z) / Ns)
    R22[-1, :] *= weight

    # right-hand side
    B = np.zeros(r22_nrows + 1)
    B[-1] = weight * Ns
    x, residuals, rnk, s_ = np.linalg.lstsq(R22, B, rcond=None)
    c_res = x[:-1]
    d_res = x[-1]

    # check if d_res is suited for zeros calculation
    if np.abs(d_res) < tol_res:
        # d_res is too small, discard solution and proceed the |d_res| = tol_res
        # print(f'Replacing d_res solution as it was too small ({d_res}).')
        d_res = tol_res if d_res > 0 else -tol_res

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

    return poles

def vector_fit(freq, 
               Z, 
               init_poles, 
               max_iter=100, 
               tol=1e-8, 
               atol=1e-8,
               rtol=1e-8,
               d_fit=True, 
               h_fit=True,
               auto_scale=True,
               #only_poles=False,
               method="pole_relocating"
               ) -> Tuple[np.ndarray, np.ndarray, float, float]:
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
    - # only_poles: Whether to only fit the poles.
    
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

    if method == "SimonAFS":
        max_iter = 1

    for _ in range(max_iter):

        poles = sort_poles_residues(poles, atol=atol, rtol=rtol)
        cindex = get_cindex(poles, atol=atol, rtol=rtol)

        # pole_relocatin method
        if method == "pole_relocating":
            poles = _pole_relocating_vector_fit(s, Z, d_fit, h_fit, Ns, N_poles, poles, cindex, tol_res=tol)
        elif method == "Gustavsen":
            poles = _Gustavsen_vector_fit(s, Z, d_fit, h_fit, Ns, N_poles, poles, cindex)
        elif method == "SimonAFS":
            return construct_AB_matrices(s, Z, d_fit, h_fit, Ns, N_poles, poles, cindex)
        else:
            raise ValueError(f"Invalid method: {method}")
        
    poles = sort_poles_residues(poles)
    
    # stage 2: residue identification
    A = np.zeros((Ns, N_poles + 2), dtype=complex)
    # [c1, c2, ..., cn, d, e]
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1 / (s - p)
        elif cindex[i] == 1:
            A[:, i] = 1 / (s - p) + 1 / (s - np.conj(p))
        elif cindex[i] == 2:
            A[:, i] = 1j / (s - p) - 1j / (s - np.conj(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

    if d_fit:
        A[:, N_poles] = 1
    if h_fit:
        A[:, N_poles + 1] = s
    B = Z

    A = np.vstack([A.real, A.imag])
    B = np.hstack([B.real, B.imag])

    x, residuals, rnk, s_ = np.linalg.lstsq(A, B, rcond=None)

    residues = np.zeros(N_poles, dtype=np.complex64)
    for i, ci in enumerate(cindex):
        if ci == 0:
            residues[i] = x[i]
        elif ci == 1:
            rv, iv = x[i], x[i + 1]
            residues[i] = rv - 1j * iv
            residues[i + 1] = rv + 1j * iv

    if d_fit:
        d = np.real(x[N_poles])
    if h_fit:
        h = np.real(x[N_poles + 1])

    if auto_scale:
        s = s * s_scale
        poles = poles * s_scale
        residues = residues * s_scale
        if h_fit:
            h = h / s_scale
    
    if VF_DEBUG:
        print_params(cindex, poles, residues, d, h)
        Z_fit = evaluate(poles, residues, d, h, s)
        rms_error = fitted_error_estimate(Z, Z_fit, method="RMSE")
        print(f'RMSE: {rms_error:.2E}')
    return poles, residues, d, h

