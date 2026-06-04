import numpy as np
import matplotlib.pyplot as plt

def lorenz(t, state, sigma = 10, rho = 28, beta = 8/3):
    x, y, z = state
    xt = sigma*(y - x)
    yt = x*(rho - z) - y
    zt = x*y - beta*z
    return [xt, yt, zt]

def rossler(t, state, a = 0.3, b = 0.2, c = 5.7, drive = 0):
    x, y, z = state
    xt = -y - z + drive # adding lorenz x(t) here
    yt = x + a*y
    zt = b + z*(x - c)
    return [xt, yt, zt]

# here we are simulating the Lorenz system and generating states
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
lx = (lx - np.mean(lx))/np.std(lx) # normalised as getting the 'SVD did not converge error' -- numerical instability during pseudoinv

lz = l_states[:, 2]
lz_raw = lz.copy()  # unnormalised z(t)
lz_mean, lz_std = np.mean(lz), np.std(lz)
lz = (lz - lz_mean)/lz_std

# reservoir -- we do the same thing here too
ross_state = np.array([1.0, 1.0, 1.0])
# ross_state = np.random.rand(3) # --> randomising it instead for more complexity
ross_res = [] # essentially S_i

for i in range(T):
    dx, dy, dz = rossler(t[i], ross_state, drive = lx[i])
    ross_state += dt * np.array([dx, dy, dz])
    ross_res.append(ross_state.copy())

ross_res = np.array(ross_res)

# memory extension
m = 1200
w = np.exp(-np.linspace(0, 5, m + 1)) # w_0 to w_m exponentially decaying

X_i = []

for i in range(m, T):
    stack = []
    for j in range(m + 1):
        past = ross_res[i - m + j] # S_(i - m + j) -- past state
        weighted = w[j] * past
        stack.append(weighted)
    
    X_i.append(np.concatenate(stack)) # convert to 1D

X_i = np.array(X_i).T
X_aug = np.vstack([X_i, X_i**2, X_i**3])

# readout training using Moore-Penrose pseudoinverse
v = lz[m:].reshape(1, -1) # targets in a 2D row vector with shape (1, len(lz))
weights = v @ np.linalg.pinv(X_aug)

prediction = weights @ X_aug

# unnormalising
prediction_real = prediction * lz_std + lz_mean

rmse = np.sqrt(np.mean((prediction_real.flatten() - lz_raw[m:])**2))

plt.plot(t[m:], lz_raw[m:], label = 'Truth Values')
plt.plot(t[m:], prediction_real.flatten(), '--', label = f'Predicted Values; RMSE = {round(rmse, 4)}')
plt.title('Rossler RC to Predict Lorenz z(t)')
plt.xlabel('t')
plt.ylabel('z(t)')
plt.legend()
plt.grid(True)
plt.show()