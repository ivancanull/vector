from math import e
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def rational_func(s, poles, residues, d=0):
    return np.sum(residues / (s[:, None] - poles), axis=1) + d

def residuals(params, s, H, num_poles):

    # Extract poles and residues from params
    poles = params[:num_poles]
    residues = params[num_poles:]
    
    # Compute rational function
    H_fitted = rational_func(s, poles, residues, d=0)
    
    # Compute residuals
    return np.abs(H - H_fitted)

"""
Step 1: Generate Dummy Data
"""
# Generate frequency range
f = np.linspace(1, 1e6, 500)
s = 1j * 2 * np.pi * f

# True system response (for example purposes)
H_true = (2 * np.pi * 1e5) / (s + 2 * np.pi * 1e5)


"""
Step 2: Perform Vector Fitting
"""

# Initial guess for poles and residues
initial_poles = np.array([-2 * np.pi * 10])
initial_residues = np.array([-2 * np.pi * 10])
num_poles = len(initial_poles)

# Combine initial poles and residues into a single parameter array
initial_params = np.hstack([initial_poles.view(np.float64), initial_residues.view(np.float64)])

# Perform least squares fitting
result = least_squares(residuals, initial_params, args=(s, H_true, num_poles)) # Crazy

# Extract refined poles and residues
refined_poles = result.x[:num_poles]
refined_residues = result.x[num_poles:]

print("Refined Poles:", refined_poles)
print("Refined Residues:", refined_residues)

# Compute fitted response
H_fitted = rational_func(s, refined_poles, refined_residues)

plt.plot(f, H_true, label='True Response')
plt.plot(f, H_fitted, label='Fitted Response', linestyle='dashed')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.savefig('demo_vf_fitting_single.png')
