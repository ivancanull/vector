from typing import Tuple
import re
import numpy as np
from .utils import frequency_to_s, get_cindex, is_real
from .error_estimate import poles_residues_error_estimate, fitted_error_estimate, poles_sorting
from .complex_func import FunctionAbstract

VF_DEBUG = False
__all__ = ["VectorFitting", "vector_fit", "evaluate"]

class VectorFitting:
    def __init__(self, freq, Z, init_poles, max_iter=100, tol=1e-6, relaxed=False, d_fit=True, h_fit=True):
        """
        Initialize the VectorFitting instance.
        
        Parameters:
        - freq: An array of frequency values (Hz).
        - Z: An array of complex f(s) values at the frequency points | ComplexFunction instance which represemts f(s).
        - init_poles: Initial poles provided by the user.
        - max_iter: The maximum number of iterations for the fitting process.
        - tol: The convergence tolerance.
        - relaxed: Whether to use d residues in Gustavsen's relocating method.

        References:
        - Gustavsen, Bjorn, and Adam Semlyen. "Rational approximation of frequency domain responses by vector fitting." IEEE Transactions on power delivery 14.3 (1999): 1052-1061.

        - Gustavsen, Bjørn. "Improving the pole relocating properties of vector fitting." IEEE Transactions on Power Delivery 21.3 (2006): 1587-1592.

        """
        self.freq = freq
        self.s = frequency_to_s(freq)
        # if Z is a given callable ComplexFunction instance, evaluate it at the frequency points
        if isinstance(Z, FunctionAbstract):
            self.Z = Z(self.s)
        else:
            self.Z = Z
        
        # make sure Z is the same length as freq
        if len(self.Z) != len(freq):
            raise ValueError("Z must be the same length as freq")
        
        # responses will be weighted according to their norm;
        # alternative: equal weights with weight_response = 1.0
        # or anti-proportional weights with weight_response = 1 / np.linalg.norm(freq_response)
        # weights_responses = np.linalg.norm(self.Z)
        # weights_responses = np.ones(self.network.nports ** 2)
        # weights_responses = 10 / np.exp(np.mean(np.log(np.abs(freq_responses)), axis=1))
        
        self.init_poles = np.array(init_poles)
        self.max_iter = max_iter
        self.tol = tol
        self.N_poles = len(init_poles)
        self.poles = self.init_poles.copy()
        self.residues = np.zeros_like(self.init_poles, dtype=complex)
        self.d = 0
        self.h = 0

        self.d_fit = d_fit
        self.h_fit = h_fit

        # whether to use d residues in Gustavsen's relocating method
        self.relaxed = relaxed

        self.true_poles_set = False

    def set_true_poles_residues(self, poles, residues, d, h):
        """
        Set the true poles, residues, d, and h values for error calculation.
        """
        self.true_poles = poles
        self.true_residues = residues
        self.true_d = d
        self.true_h = h
        self.true_poles_set = True

    def estimate_poles_residues_error(self, method="Error"):
        """
        Estimate the error between the true and fitted poles and residues. The error calculation method supports:
        - Error: Direct error calculation.
        
        Parameters:
        - method: The error calculation method.
        
        Returns:
        - The error estimate. a, b, c, d.
        """
        # check if the true poles and residues are set
        if not hasattr(self, "true_poles"):
            raise ValueError("True poles and residues are not set.")
        return poles_residues_error_estimate(
            self.true_poles, self.true_residues, self.true_d, self.true_h, 
            self.poles, self.residues, self.d, self.h, method)
            
    def _get_cindex(self, poles):
        """
        Get the cindex array for the given poles. Assert that complex poles come in real values or conjugate pairs.
        """
        N = len(poles)
        cindex = np.zeros(N)
        
        for i, p in enumerate(poles):
            if p.imag != 0:
                if i == 0 or cindex[i-1] != 1:
                    assert np.conj(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % (poles[i], poles[i+1]))
                    cindex[i] = 1
                else:
                    cindex[i] = 2
        return cindex
        
    def fit(self):
        """
        Fit the rational function to the given data.
        """
        
        for iteration in range(self.max_iter):
            
            self.cindex = get_cindex(self.poles)

            # stage 1: poles and zeros identification
            if self.relaxed:
                self._pole_relocating(self.s)
                self._fit_residues(self.s)
            else:
                # appendix A: pole identification
                A, B = self._construct_AB_matrices(self.s, stage="pole")
                x, residuals, rnk, s_ = np.linalg.lstsq(A, B, rcond=None)
                # Appendix B: calculation of zeros
                self._calculate_zeros(x[self.N_poles + 2:])

                # stage 2: residue identification
                self.cindex = get_cindex(self.poles)
                A, B = self._construct_AB_matrices(self.s, stage="residue")
                
                x, residuals, rnk, s_ = np.linalg.lstsq(A, B, rcond=1e-10)
                self._update_parameters_residue(x)

            if VF_DEBUG:
                print('-' * 50)
                print(f'Iteration {iteration + 1}')
                if self.true_poles_set:
                    sorted_poles, sorted_residues, sorted_poles_fit, sorted_residues_fit = self.estimate_poles_residues_error(method="Sort")
                    self.print_params(sorted_poles, sorted_residues, self.true_d, self.true_h)
                    self.print_params(sorted_poles_fit, sorted_residues_fit, self.d, self.h, postfix="fit")
                else:
                    # sort the poles and residues in ascending order of real part
                    sorted_poles_fit, sorted_residues_fit = poles_sorting(self.poles, self.residues)
                    self.print_params(sorted_poles_fit, sorted_residues_fit, self.d, self.h, postfix="fit")
                rms_error = fitted_error_estimate(self.Z, self.evaluate(self.s), method="RMSE")
                print(f'RMSE: {rms_error:.2E}')

                # if len(sorted_poles) == len(sorted_poles_fit):
                #     poles_error, residues_error, d_error, h_error = self.estimate_poles_residues_error(method="Error")
                #     self.print_params(poles_error, residues_error, d_error, h_error, postfix="error")
    
    def _pole_relocating(self, s):
        """
        Relocate the poles of the rational function.
        """
        Ns = len(s)
        A = np.zeros((Ns, self.N_poles * 2 + 3), dtype=complex)
        # [c1, c2, ..., cn, d, e, c1_res, c2_res, ..., cn_res, d_res]
        for i, p in enumerate(self.poles):
            if self.cindex[i] == 0:
                A[:, i] = 1 / (s - p)
            elif self.cindex[i] == 1:
                A[:, i] = 1 / (s - p) + 1 / (s - np.conj(p))
            elif self.cindex[i] == 2:
                A[:, i] = 1j / (s - p) - 1j / (s - np.conj(p))
            else:
                raise RuntimeError("cindex[%s] = %s" % (i, self.cindex[i]))

            A[:, self.N_poles + 2 + i] = -A[:, i] * self.Z

        A[:, self.N_poles] = 1
        A[:, self.N_poles + 1] = s
        A[:, self.N_poles * 2 + 2] = -1 * self.Z

        # sum up the real of the poles for future use
        trival_coeff = np.sum(A[:, :self.N_poles].real, axis=0)

        A = np.vstack([A.real, A.imag])
        R = np.linalg.qr(A, mode='r')
        dim_k = min(2 * Ns, self.N_poles * 2 + 3)
        if dim_k < self.N_poles * 2 + 3:
            r12_nrows = Ns
            r22_nrows = Ns
        else:
            r12_nrows = self.N_poles + 2
            r22_nrows = self.N_poles + 1
        r12_ncols = self.N_poles + 2
        r22_ncols = self.N_poles + 1

        # restrict the d_res to be non-zero
        R22 = R[r12_nrows:, r12_ncols:]
        
        R22 = np.vstack([R22, np.zeros((1, r22_ncols))])
        R22[-1, :-1] = np.real(trival_coeff)
        R22[-1, -1] = Ns

        # weight the last d_res LS problem
        weight = np.sqrt(np.linalg.norm(self.Z) / Ns)
        R22[-1, :] *= weight

        # right-hand side
        B = np.zeros(r22_nrows + 1)
        B[-1] = weight * Ns
        x, residuals, rnk, s_ = np.linalg.lstsq(R22, B, rcond=None)
        c_res = x[:-1]
        d_res = x[-1]

        # check if d_res is suited for zeros calculation
        tol_res = 1e-8
        if np.abs(d_res) < tol_res:
            # d_res is too small, discard solution and proceed the |d_res| = tol_res
            print(f'Replacing d_res solution as it was too small ({d_res}).')
            d_res = tol_res if d_res > 0 else -tol_res

        # build test matrix H
        H = np.zeros((self.N_poles, self.N_poles), dtype=complex)
        for i, (ci, p) in enumerate(zip(self.cindex, self.poles)):
            if ci == 0:
                H[i, i] = p
                H[i] -= c_res / d_res
            elif ci == 1:
                x, y = np.real(p), np.imag(p)
                H[i, i] = H[i + 1, i + 1] = x
                H[i, i + 1] = -y
                H[i + 1, i] = y
                H[i] -= 2 * c_res / d_res

        self.poles = np.linalg.eigvals(H)
        sorted_indices = np.argsort(self.poles, kind='mergesort')
        self.poles = self.poles[sorted_indices]
        unstable = np.real(self.poles) > 0
        self.poles[unstable] -= 2 * np.real(self.poles)[unstable]
        
        return

    def _fit_residues(self, s):
        """
        Fit residues of the rational function.
        """

        Ns = len(s)
        A = np.zeros((Ns, self.N_poles + 2), dtype=complex)
        # [c1, c2, ..., cn, d, e]
        for i, p in enumerate(self.poles):
            if self.cindex[i] == 0:
                A[:, i] = 1 / (s - p)
            elif self.cindex[i] == 1:
                A[:, i] = 1 / (s - p) + 1 / (s - np.conj(p))
            elif self.cindex[i] == 2:
                A[:, i] = 1j / (s - p) - 1j / (s - np.conj(p))
            else:
                raise RuntimeError("cindex[%s] = %s" % (i, self.cindex[i]))


        if self.d_fit:
            A[:, self.N_poles] = 1
        if self.h_fit:
            A[:, self.N_poles + 1] = s
        B = self.Z

        A = np.vstack([A.real, A.imag])
        B = np.hstack([B.real, B.imag])

        x, residuals, rnk, s_ = np.linalg.lstsq(A, B, rcond=None)

        self.residues = np.zeros(self.N_poles, dtype=np.complex64)
        for i, ci in enumerate(self.cindex):
            if ci == 0:
                self.residues[i] = x[i]
            elif ci == 1:
                rv, iv = x[i], x[i + 1]
                self.residues[i] = rv - 1j * iv
                self.residues[i + 1] = rv + 1j * iv

        if self.d_fit:
            self.d = np.real(x[self.N_poles])
        if self.h_fit:
            self.h = np.real(x[self.N_poles + 1])
        return



    def _construct_AB_matrices(self, s, stage="pole"):
        """
        Construct the A and B matrices for the linear system Ax = B, ensuring that poles and residues are conjugate pairs.
        """
        Ns = len(s)
        if stage == "pole":
            A = np.zeros((Ns, self.N_poles * 2 + 2), dtype=complex)
        else: # stage == "residue"
            A = np.zeros((Ns, self.N_poles + 2), dtype=complex)
            
        for i, p in enumerate(self.poles):
            if self.cindex[i] == 0:
                A[:, i] = 1 / (s - p)
            elif self.cindex[i] == 1:
                A[:, i] = 1 / (s - p) + 1 / (s - np.conj(p))
            elif self.cindex[i] == 2:
                A[:, i] = 1j / (s - p) - 1j / (s - np.conj(p))
            else:
                raise RuntimeError("cindex[%s] = %s" % (i, self.cindex[i]))
            
            # -f(sk) / (sk - ai)
            if stage == "pole":
                A[:, self.N_poles + 2 + i] = -A[:, i] * self.Z
            
        A[:, self.N_poles] = 1
        A[:, self.N_poles + 1] = s

        B = self.Z
        A = np.vstack([A.real, A.imag])
        B = np.hstack([B.real, B.imag])
        print(A)

        return A, B
    
    # def _update_parameters_pole(self, x):
    #     """
    #     Update poles, residues, d, and h from the solution vector x on stage 1.
    #     """
    #     # self.residues = x[:self.N_poles]
    #     # self.d = x[self.N_poles]
    #     # self.h = x[self.N_poles + 1]
    
    def _update_parameters_residue(self, x):
        """
        Update poles, residues, d, and h from the solution vector x on stage 2.
        """

        self.residues = np.zeros(self.N_poles, dtype=np.complex64)
        for i, ci in enumerate(self.cindex):
            if ci == 0:
                self.residues[i] = x[i]
            elif ci == 1:
                rv, iv = x[i], x[i + 1]
                self.residues[i] = rv - 1j * iv
                self.residues[i + 1] = rv + 1j * iv
            else:
                pass

        self.d = np.real(x[self.N_poles])
        self.h = np.real(x[self.N_poles + 1])

    def _calculate_zeros(self, C):
        """
        Calculate the zeros of the rational function.
        """
        A = np.diag(self.poles)
        B = np.ones(self.N_poles)
        for i, (ci, p) in enumerate(zip(self.cindex, self.poles)):
            if ci == 1:
                x, y = np.real(p), np.imag(p)
                A[i, i] = A[i + 1, i + 1] = x
                A[i, i + 1] = -y
                A[i + 1, i] = y
                B[i] = 2
                B[i + 1] = 0
        
        H = np.real(A - np.outer(B, C))
        self.poles = np.linalg.eigvals(H)
        unstable = np.real(self.poles) > 0
        self.poles[unstable] -= 2 * np.real(self.poles)[unstable]
        return 

    def print_poles_residues(self, poles, residues, postfix=""):
        print(f'Poles {postfix}\t\t\t\tResidues {postfix}')
        cindex = get_cindex(poles)
        for i, (p, r) in enumerate(zip(poles, residues)):
            if cindex[i] == 0:

                print(f'{p.real:.2E}\t\t\t{r.real:.2E}')
            elif cindex[i] == 1:
                print(f'{p.real:.2E} +- {abs(p.imag):.2E}j\t\t{r.real:.2E} +- {abs(r.imag):.2E}j\t')
            else:
                pass

    def print_params(self, poles, residues, d, h, postfix=""):
        """
        Print the poles, residues, d, and h values. 
        """
        # cfmt = lambda x: f'{x.real:.2E} {"+-"[x.imag > 0]} {abs(x.imag):.2E}j'
        self.print_poles_residues(poles, residues, postfix)
        print ("d " + postfix + ": " + "{:.2E}".format(d))
        print ("h " + postfix + ": " + "{:.2E}".format(h))
    
    def evaluate(self, s):
        """
        Evaluate the rational function at given complex frequency points s.
        
        Parameters:
        - s: A complex number or array of complex numbers where the function is to be evaluated.
        
        Returns:
        - The value of the rational function at s or an array of values if s is an array.
        """
        f = self.d + self.h * s

        for i in range(self.N_poles):
            f += self.residues[i] / (s - self.poles[i])
        return f

def get_cindex(poles, atol=1e-12, rtol=1e-12):
    """
    Get the cindex array for the given poles. Assert that complex poles come in real values or conjugate pairs.
    """
    N = len(poles)
    cindex = np.zeros(N)
    i = 0
    while i < N:
        if not is_real(poles[i], atol=atol, rtol=rtol):
            assert np.allclose(np.conj(poles[i]), poles[i+1]) , ("Complex poles must come in conjugate pairs: %s, %s" % (poles[i], poles[i+1]), '\n', 'All poles:', poles)
            cindex[i] = 1
            cindex[i+1] = 2
            i += 2
        else:
            i += 1
    return cindex

def sort_poles_residues(poles, residues=None, atol=1e-12, rtol=1e-12):
    """
    Sort the poles and residues in ascending order of real part.
    """
    # put the real poles first
    condition = is_real(poles, atol=atol, rtol=rtol)
    # sort the real poles
    real_poles = poles[condition].real # force the poles to be real
    complex_poles = poles[~condition]
    if residues is not None:
        real_residues = residues[condition]
        complex_residues = residues[~condition]
    
    # sort the complex poles by real part
    real_poles_idx = np.argsort(np.abs(real_poles))
    complex_poles_idx = np.argsort(np.abs(complex_poles))
    real_poles = real_poles[real_poles_idx]
    complex_poles = complex_poles[complex_poles_idx]

    if residues is not None:
        real_residues = real_residues[real_poles_idx]
        complex_residues = complex_residues[complex_poles_idx]
        return np.concatenate([real_poles, complex_poles]), np.concatenate([real_residues, complex_residues])
    else:
        return np.concatenate([real_poles, complex_poles])

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

def construct_AB_matrices(s, cindex, poles, Z):
    """
    Construct the A and B matrices for the linear system Ax = B, ensuring that poles and residues are conjugate pairs.
    """
    Ns = len(s)
    N_poles = len(poles)

    A = np.zeros((Ns, N_poles * 2 + 2), dtype=complex)
        
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1 / (s - p)
        elif cindex[i] == 1:
            A[:, i] = 1 / (s - p) + 1 / (s - np.conj(p))
        elif cindex[i] == 2:
            A[:, i] = 1j / (s - p) - 1j / (s - np.conj(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))
        
        # -f(sk) / (sk - ai)
        A[:, N_poles + 2 + i] = -A[:, i] * Z
        
    A[:, N_poles] = 1
    A[:, N_poles + 1] = s

    B = Z
    A = np.vstack([A.real, A.imag])
    B = np.hstack([B.real, B.imag])

    # # sum up the real of the poles for future use
    # trival_coeff = np.sum(A[:, :N_poles].real, axis=0)

    # Q, R = np.linalg.qr(A, mode='qr')
    # dim_k = min(2 * Ns, N_poles * 2 + 2)
    # if dim_k < N_poles * 2 + 2:
    #     r12_nrows = Ns
    #     r22_nrows = Ns
    # else:
    #     r12_nrows = N_poles + 2
    #     r22_nrows = N_poles
    # r12_ncols = N_poles + 2
    # r22_ncols = N_poles

    # # restrict the d_res to be non-zero
    # R22 = R[r12_nrows:, r12_ncols:]
        
    # R22 = np.vstack([R22, np.zeros((1, r22_ncols))])
    # R22[-1, :-1] = np.real(trival_coeff)
    # R22[-1, -1] = Ns

    # # weight the last d_res LS problem
    # weight = np.sqrt(np.linalg.norm(Z) / Ns)
    # R22[-1, :] *= weight

    return A, B

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

def vector_fit(freq, 
               Z, 
               init_poles, 
               max_iter=100, 
               tol=1e-8, 
               atol=1e-12,
               rtol=1e-12,
               d_fit=True, 
               h_fit=True,
               auto_scale=True,
               only_poles=False) -> Tuple[np.ndarray, np.ndarray, float, float]:
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

    for _ in range(max_iter):

        poles = sort_poles_residues(poles, atol=atol, rtol=rtol)
        cindex = get_cindex(poles, atol=atol, rtol=rtol)

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

    if only_poles:
        return poles, x, R22, B
    
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

     


        


    
    



                