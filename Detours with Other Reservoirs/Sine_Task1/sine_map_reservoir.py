import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

N_steps, n = 100000, 300 
transients = 1000
r_min, r_max = 0.86, 1 # chaotic regime

x = np.linspace(-4, 4, 500) # inputs for polynomial
x_norm = (x + 4) / 8  # maps [-4, 4] to [0, 1] as domain of our reservoir, the sine map, is [0, 1]

targets = (x - 3)*(x - 2)*(x - 1)*x*(x + 1)*(x + 2)*(x + 3) # we want to solve a 7th degree polynomial

r_enc = r_min + (r_max - r_min) * x_norm # rate encoding

def sine(r, x):
    return r*np.sin(np.pi*x)

def states(r_enc):
    X_i = [] # collection of reservoir states S_i
    for r in r_enc:
        x = np.random.rand() # initial condition
        trajectory = []
        for i in range(N_steps):
            x = sine(r, x)
            if i >= transients:
                trajectory.append(x)
        S_i = trajectory[:n]
        X_i.append(S_i)

    return X_i

X_i = np.array(states(r_enc)).T
print('Reservoir Dimensions: ', X_i.shape)

std = np.std(X_i, axis = 1, keepdims = True)
std[std == 0] = 1

X_i = (X_i - np.mean(X_i, axis = 1, keepdims = True))/std # normalising states
X_aug = np.vstack([X_i, X_i**2]) # making the readout layer nonlinear

# readout training using Moore-Penrose pseudoinverse
v = targets.reshape(1, -1)
w = v @ np.linalg.pinv(X_aug)

prediction = w @ X_aug

# RMSE
rmse = np.sqrt(np.mean((prediction.flatten() - targets)**2))

plt.plot(x, targets, label = 'Truth Values')
plt.plot(x, prediction.flatten(), '--', label = f'Predicted Values; RMSE = {round(rmse, 4)}')
plt.title('Sine Map RC')
plt.xlim(-3, 3)
plt.ylim(-120, 120)
plt.legend()
plt.grid(True)
plt.show()

#---finding roots---
x_val = x
y_val = prediction.flatten() # making it 1D

f_interpol = interp1d(x_val, y_val, kind = 'linear', fill_value = 'extrapolate') # interpolating as previously we had discrete values: linear spline interpol

def int_pred(x):
    return f_interpol(x) # our predicted function

guess = np.linspace(-4, 4, 9) # initial guesses for x 
'''fsolve is a local root finder so the number and the accuracy of roots depends on the number of initial guesses.
This map seemed too sensitive to initial guesses'''
roots = fsolve(int_pred, guess)

def unique(roots, tol = 1e-2): # for unique roots
    # for each root, if it's far enough from the last one (> tolerance), it is added to the unique list
    roots = np.sort(roots)
    unique = []
    for r in roots:
        if not unique or abs(r - unique[-1]) > tol:
            unique.append(r)
    return np.array(unique)

final_roots = unique(roots)
print('Roots: ', np.round(final_roots, 4))