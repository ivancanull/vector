from matplotlib import lines, tight_layout
import numpy as np
import matplotlib.pyplot as plt

from src import *

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

# Define the function and frequency sample points
response = ComplexFunction(poles, residues, d, h)
freq = np.linspace(0, 2e4, 1000)

# Evaluate with a 2D array input
s = frequency_to_s(freq)
fs_value = response(s)

# Initialize and fit VectorFitting instance

init_poles = [
    -1e-2+1j, -1e-2-1j,
    -111+11100j, -111-11100j,
    -222+22200j, -222-22200j,
    -333+33300j, -333-33300j,
    -444+44400j, -444-44400j,
    -555+55500j, -555-55500j,
    -666+66600j, -666-66600j,
    -777+77700j, -777-77700j,
    -888+88800j, -888-88800j,
    -1000+100000j, -1000-100000j
]

vf = VectorFitting(freq, response, init_poles, max_iter=2, tol=1e-6)
vf.set_true_poles_residues(poles, residues, d, h)

vf.fit()
fitted_Z = vf.evaluate(s)


fig, axes = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)
axes[0].plot(freq, np.real(fs_value), label='True Response', marker='o')
axes[0].plot(freq, np.real(fitted_Z), label='Fitted Response', linestyle='dashed')
axes[1].plot(freq, np.imag(fs_value), label='True Response', marker='o')
axes[1].plot(freq, np.imag(fitted_Z), label='Fitted Response', linestyle='dashed')
axes[2].plot(freq, 20 * np.log10(np.abs(fs_value)), label='True Response', marker='o')
axes[2].plot(freq, 20 * np.log10(np.abs(fitted_Z)), label='Fitted Response', linestyle='dashed')
for ax in axes:
    ax.set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Real')
axes[1].set_ylabel('Imag')
axes[2].set_ylabel("Magnitude (dB)")

plt.legend()
plt.savefig("test_vf.pdf")

