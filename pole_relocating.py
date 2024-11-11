import numpy as np
import matplotlib.pyplot as plt

from src import *

# Example 1

poles = [
    -2 * np.pi * 1e5,
]
residues = [
    2 * np.pi * 1e5,
]
d = 0
h = 0

# Define the function and frequency sample points
response = ComplexFunction(poles, residues, d, h)
freq = np.logspace(1, 6, num=2000)

# Evaluate with a 2D array input
s = frequency_to_s(freq)
fs_value = response(s)

# Initialize and fit VectorFitting instance

init_poles = [
    -2 * np.pi * 10,
    #  -2 * np.pi * 20,
]

vf = VectorFitting(freq, response, init_poles, max_iter=2, tol=1e-6)
vf.set_true_poles_residues(poles, residues, d, h)
vf.fit()
fitted_Z = vf.evaluate(s)
f = np.zeros_like(s) 
for i in range(vf.N_poles):
    f += vf.residues[i] / (s - vf.poles[i])

print()
print()
vf_relaxed = VectorFitting(freq, response, init_poles, max_iter=2, tol=1e-6, relaxed=True, d_fit=False, e_fit=False)
vf_relaxed.set_true_poles_residues(poles, residues, d, h)
vf_relaxed.fit()
fitted_Z_relaxed = vf_relaxed.evaluate(s)

fig, axes = plt.subplots()
axes.plot(freq, np.abs(fs_value), label='True Response')
axes.plot(freq, np.abs(f), label='Sigma (s)')
axes.plot(freq, np.abs(fitted_Z), label='Fitted Response')
axes.plot(freq, np.abs(fitted_Z_relaxed), label='Fitted Response (Relaxed)')

axes.set_xlabel('Frequency (Hz)')
axes.set_ylabel("Magnitude")
axes.set_xscale('log')
axes.set_yscale('log')
plt.legend()
plt.savefig("pole_relocating.png")

