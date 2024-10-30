from re import DEBUG
import numpy as np
from .utils import frequency_to_s, get_cindex
from .error_estimate import poles_residues_error_estimate, fitted_error_estimate
from .complex_func import ComplexFunction

DEBUG = True
__all__ = ["VectorFitting"]

class VectorFitting:
    def __init__(self, freq, Z, init_poles, max_iter=100, tol=1e-6, relaxed=False):
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
        if isinstance(Z, ComplexFunction):
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

        # whether to use d residues in Gustavsen's relocating method
        self.relaxed = relaxed

    def set_true_poles_residues(self, poles, residues, d, h):
        """
        Set the true poles, residues, d, and h values for error calculation.
        """
        self.true_poles = poles
        self.true_residues = residues
        self.true_d = d
        self.true_h = h

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
            
            
            # Stage 1: poles and zeros identification

            # Appendix A: pole identification
            self.cindex = get_cindex(self.poles)
            A, B = self._construct_AB_matrices(self.s, stage="pole")
            x, residuals, rnk, s_ = np.linalg.lstsq(A, B, rcond=None)
            print(x)
            # Appendix B: calculation of zeros
            self._calculate_zeros(x[self.N_poles + 2:])

            # Stage 2: residue identification
            
            self.cindex = get_cindex(self.poles)
            A, B = self._construct_AB_matrices(self.s, stage="residue")
            
            x, residuals, rnk, s_ = np.linalg.lstsq(A, B, rcond=None)
            self._update_parameters_residue(x)

            if DEBUG:
                print('-' * 50)
                print(f'Iteration {iteration + 1}')
                sorted_poles, sorted_residues, sorted_poles_fit, sorted_residues_fit = self.estimate_poles_residues_error(method="Sort")
                self.print_params(sorted_poles, sorted_residues, self.true_d, self.true_h)
                self.print_params(sorted_poles_fit, sorted_residues_fit, self.d, self.h, postfix="fit")
                rms_error = fitted_error_estimate(self.Z, self.evaluate(self.s), method="RMSE")
                print(f'RMSE: {rms_error:.2E}')

                # if len(sorted_poles) == len(sorted_poles_fit):
                #     poles_error, residues_error, d_error, h_error = self.estimate_poles_residues_error(method="Error")
                #     self.print_params(poles_error, residues_error, d_error, h_error, postfix="error")
            
    
    def _construct_AB_matrices(self, s, stage="pole"):
        """
        Construct the A and B matrices for the linear system Ax = B, ensuring that poles and residues are conjugate pairs.
        """
        Ns = len(s)
        if stage == "pole":
            if self.relaxed:
                A = np.zeros((Ns, self.N_poles * 2 + 3), dtype=complex)
            else:
                A = np.zeros((Ns, self.N_poles * 2 + 2), dtype=complex)
        else: # stage == "residue"
            A = np.zeros((Ns, self.N_poles + 2), dtype=complex)

        if self.relaxed and stage == "pole":
            weight = np.sqrt(np.linalg.norm(self.Z) / Ns)
            # print(f'Weight: {weight}')
            weight = 1

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

        if self.relaxed and stage == "pole":
            A[:, self.N_poles + 2] = -weight * self.Z

        B = self.Z
        A = np.vstack([A.real, A.imag])
        B = np.hstack([B.real, B.imag])

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

        if self.relaxed:
            d_res = C[-1]
            # check if d_res is suited for zeros calculation
            tol_res = 1e-8
            if np.abs(d_res) < tol_res:
                # d_res is too small, discard solution and proceed the |d_res| = tol_res
                print(f'Replacing d_res solution as it was too small ({d_res}).')
                d_res = tol_res if d_res > 0 else -tol_res
            print(f'd_res: {d_res}')
            C_res = C[:-1] / d_res
        else:
            C_res = C
        
        H = np.real(A - np.outer(B, C_res))
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
