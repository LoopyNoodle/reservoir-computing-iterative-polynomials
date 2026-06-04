import numpy as np
import matplotlib.pyplot as plt

# applying the same things we did in logistic here for sine map (like that reduction to stats thing)

def sine(r, x):
    return r*np.sin(np.pi*x)

def lorenz(t, state, sigma = 10, rho = 28, beta = 8/3):
    x, y, z = state
    xt = sigma*(y - x)
    yt = x*(rho - z) - y
    zt = x*y - beta*z
    return [xt, yt, zt]

# simulating the Lorenz system and generating states
dt = 0.01 # time step as we are using euler's method: x_(n+1) = x_n + derivative * dt
T = 10000 # total time steps
t = np.linspace(0, T*dt, T) # time values for each step

state = np.array([1.0, 1.0, 1.0]) # initial conditions
l_states = [] # all states in the lorenz system as the system evolves -- our time series

# the actual application of euler
for i in range(T):
    dx, dy, dz = lorenz(t[i], state) # calculating our derivatives for euler method
    state += dt * np.array([dx, dy, dz])
    l_states.append(state.copy())

l_states = np.array(l_states) # each row is one time step and each column is an x, y, z value

lx = l_states[:, 0]
lz = l_states[:, 2]
lz_raw = lz.copy()

lx = (lx - np.mean(lx))/np.std(lx) 
'''normalised as getting the 'SVD did not converge error' -- numerical instability during pseudoinv'''
lz_mean, lz_std = np.mean(lz), np.std(lz)
lz = (lz - lz_mean)/lz_std

transients, N_steps = 500, 10000
r_min, r_max = 0.86, 1 # chaotic regime

lx_min, lx_max = np.min(lx), np.max(lx)
r_enc = r_min + (r_max - r_min) * ((lx - lx_min)/(lx_max - lx_min))

X = [] # logistic reservoir state
for r in r_enc:
    x = np.random.rand() # initial state
    trajectory = []
    for i in range(N_steps):
        x = sine(r, x)
        if i >= transients:
            trajectory.append(x)
    trajectory = np.array(trajectory)
    features = [np.mean(trajectory), np.std(trajectory), np.max(trajectory), np.min(trajectory)]
    X.append(np.array(features))

# memory stacking as it is a temporal task
m = 1000
w = np.exp(-np.linspace(0, 5, m + 1)) # fading memory -- exponential decaying weights again

X_i = []
for i in range(m, T):
    stack = []
    for j in range(m + 1):
        past = X[i - m + j]
        weighted = w[j] * past
        stack.append(weighted)
    stack = np.concatenate(stack) # concatenate instead of np.array(stack)
    X_i.append(stack)

X_i = np.array(X_i).T
X_aug = np.vstack([X_i, X_i**2])

v = lz[m:].reshape(1, -1)
weights = v @ np.linalg.pinv(X_aug)

prediction = weights @ X_aug
prediction_real = prediction * lz_std + lz_mean

rmse = np.sqrt(np.mean((prediction_real.flatten() - lz_raw[m:])**2))

plt.plot(t[m:], lz_raw[m:], label = 'Truth Values')
plt.plot(t[m:], prediction_real.flatten(), '--', label = f'Predicted Values; RMSE = {round(rmse, 4)}')
plt.title('Sine Map RC to Predict Lorenz z(t)')
plt.xlabel('t')
plt.ylabel('z(t)')
plt.legend()
plt.grid(True)
plt.show()