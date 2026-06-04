import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import pandas as pd

N = 1500

x_min, x_max = -3, 3
x = np.linspace(x_min, x_max, N)

targets = (x - 3)*(x - 2)*(x - 1)*x*(x + 1)*(x + 2)*(x + 3)

def scaling(x, x_min, x_max):
    # scale input from [x_min, x_max] to [-1, 1]
    return 2 * (x - x_min) / (x_max - x_min) - 1

def chebyshev(x, M):
    N = len(x)
    T = np.zeros((M + 1, N))
    
    T[0, :] = 1.0
    T[1, :] = x
    
    for i, x_val in enumerate(x):
        A = np.array([[2*x_val, -1],
                      [1, 0]])
        
        # state vector
        z = np.array([x_val, 1.0])  # z_1 = [T_1(x), T_0(x)]
        
        for n in range(2, M + 1):
            z = A @ z
            T[n, i] = z[0]  # taking first comp
    return T

x_scaled = scaling(x, x_min, x_max)

# FINDING THE BEST M VALUE

M_values = range(5, 20)
results = []

for i in M_values:
    Phi = chebyshev(x_scaled, i)
    w = targets @ np.linalg.pinv(Phi)
    prediction = w @ Phi

    delta = (1/N) * np.sum((prediction - targets)**2)
    log_del = -np.log10(delta) if delta > 0 else np.inf

    # 8th and 9th Chebyshev Coefficients
    C_8 = w[8] if len(w) > 8 else np.nan
    C_9 = w[9] if len(w) > 9 else np.nan

    results.append({'M': i,
        'delta': delta,
        '-log10(delta)': log_del,
        'C_8': C_8,
        'C_9': C_9})
    
    print(f'M: {round(i, 2)}\n', f'Delta: {round(delta, 3)}\n', 
          f'-log10(delta): {round(log_del, 2)}\n', f'C_8: {round(C_8, 3)}\n',
          f'C_8: {round(C_8, 3)}')
    
df = pd.DataFrame(results)
df.to_csv(r'C:\Users\Sahaana V\Desktop\ASP\static_task_M_coeffs.csv', index = False)

fig, axes = plt.subplots(1, 3, figsize = (14, 10))

# delta vs M
ax1 = axes[0]
ax1.semilogy(df['M'], df['delta'], '-', linewidth = 2, markersize = 8, color = 'teal')
ax1.set_xlabel('Poly. Degree M', fontsize = 12)
ax1.set_ylabel(r'$\delta$', fontsize = 12)
ax1.axvline(x = 7, linestyle = '--', color = 'k', label = 'M = 7')
ax1.set_title('Error (Normalised) vs M', fontsize = 14)
ax1.legend()

# -log10(delta) vs M
ax2 = axes[1]
ax2.plot(df['M'], df['-log10(delta)'], '-', linewidth = 2, markersize = 8, color = 'teal')
ax2.set_xlabel('Poly. Degree M', fontsize = 12)
ax2.set_ylabel(r'$-\log_{10}(\delta)$', fontsize = 12)
ax2.axvline(x = 7, linestyle = '--', color = 'k', label = 'M = 7')
ax2.set_title(r'$-\log_{10}(\delta)$ vs M', fontsize = 14)
ax2.legend()

# PREDICTION

# feature matrix
M = 7  # polynomial degree
Phi = chebyshev(x_scaled, M)

_lambda = 1e-8

Phi_Phi_T = Phi @ Phi.T
w = targets @ np.linalg.pinv(Phi)
prediction = w @ Phi
delta = (1/N) * np.sum((prediction - targets)**2)

ax3 = axes[2]
ax3.plot(x, targets, '-', linewidth = 2, label = 'Reference Values')
ax3.plot(x, prediction, '--', linewidth= 2, label = f'Predicted Values; $\delta$ = {round(delta, 4)}, $-\log(\delta) = {round(log_del, 2)}$')
ax3.set_xlabel('x', fontsize = 12)
ax3.set_ylabel('f(x)', fontsize = 12)
ax3.set_title(f'Chebyshev Reservoir, M = {M}', fontsize= 14)
ax3.legend()

plt.tight_layout()
plt.show()

f_interp = interp1d(x, prediction, kind ='cubic', fill_value = 'extrapolate')
 
def pred_func(x_val):
    return f_interp(x_val)

initial_guess = np.linspace(x_min, x_max, 15)
roots = fsolve(pred_func, initial_guess)

def unique_roots(roots, tol = 1e-4):
    roots = np.sort(roots)
    unique = [roots[0]]
    for r in roots[1:]:
        if abs(r - unique[-1]) > tol:
            unique.append(r)
    return np.array(unique)

fin_roots = unique_roots(roots)
fin_roots = fin_roots[(fin_roots >= x_min) & (fin_roots <= x_max)]

print('Found roots:', np.round(fin_roots, 6))