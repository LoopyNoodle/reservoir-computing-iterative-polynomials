import numpy as np
import matplotlib.pyplot as plt

# take 2 for task 2: trying using duffing as rossler was giving weird outputs

def duffing(t, state, f, omega = 1.2, delta = 0.3, beta = -1.0, alpha = 1.0):
    x, y = state
    xt = y
    yt = -delta*y - beta*x - alpha*x**3 + f * np.cos(omega * t)
    return [xt, yt]

def lorenz(t, state, sigma = 10, rho = 28, beta = 8/3):
    x, y, z = state
    xt = sigma*(y - x)
    yt = x*(rho - z) - y
    zt = x*y - beta*z
    return [xt, yt, zt]

# using RK4 instead of euler for duffing
def rk4(func, t, state, dt, *args):
    k1 = np.array(func(t, state, *args))
    k2 = np.array(func(t + dt/2, state + dt * k1 / 2, *args))
    k3 = np.array(func(t + dt/2, state + dt * k2 / 2, *args))
    k4 = np.array(func(t + dt, state + dt * k3, *args))
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

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
lz = l_states[:, 2]
lz_raw = lz.copy()

lx = (lx - np.mean(lx))/np.std(lx) 
'''normalised as getting the 'SVD did not converge error' -- numerical instability during pseudoinv'''
lz_mean, lz_std = np.mean(lz), np.std(lz)
lz = (lz - lz_mean)/lz_std

lx_min, lx_max = np.min(lx), np.max(lx) # we are mapping lx to f in [1, 2] and this becomes the input drive for the duffing oscillator
lz_min, lz_max = np.min(lz), np.max(lz)
f_min, f_max = 0.20, 25.0 # [1, 2] used in the original paper
# we need a wide range for richer duffing dynamics

'''used x(t) to drive the system before -- good for inference but not forecasting'''

# u_norm = (lx - lx_min)/(lx_max - lx_min)
# f_enc = f_min + (f_max - f_min) * u_norm

'''using z(t) instead of x(t) to drive the system so that extrapolation also works'''

u_norm = (lz - lz_min)/(lz_max - lz_min)
f_enc = f_min + (f_max - f_min) * u_norm

# simulating duffing
duff_state = np.random.rand(2) # randomised initial cond
duff_res = []

'''
for i in range(T):
    dx, dy = duffing(t[i], duff_state, f = f_enc[i])
    duff_state += dt * np.array([dx, dy])
    duff_res.append(duff_state.copy())
'''

# RK4 seems to give better results than euler
for i in range(T):
    duff_state = rk4(duffing, t[i], duff_state, dt, f_enc[i])
    duff_res.append(duff_state.copy())

# memory stacking as it is a temporal task
m = 1500
w = np.exp(-np.linspace(0, 5, m + 1)) # fading memory -- exponential decaying weights again

X_i = []

for i in range(m, T):
    stack = []
    for j in range(m + 1):
        past = duff_res[i - m + j] # S_(i - m + j)
        weighted = w[j]*past
        stack.append(weighted)
    X_i.append(np.concatenate(stack))

X_i = np.array(X_i).T
X_aug = np.vstack([X_i, X_i**2, X_i**3])

v = lz[m:].reshape(1, -1)
weights = v @ np.linalg.pinv(X_aug)

prediction = weights @ X_aug
prediction_real = prediction * lz_std + lz_mean

rmse = np.sqrt(np.mean((prediction_real.flatten() - lz_raw[m:])**2))

plt.plot(t[m:], lz_raw[m:], label = 'Truth Values')
plt.plot(t[m:], prediction_real.flatten(), '--', label = f'Predicted Values; RMSE = {round(rmse, 4)}')
plt.title('Duffing RC to Predict Lorenz z(t)')
plt.xlabel('t')
plt.ylabel('z(t)')
plt.legend()
plt.grid(True)
plt.show()

# for troubleshooting (duffing output)
# if it looks like some periodic wave, then there is no chaos or complexity in dynamics
duff_arr = np.array(duff_res)
plt.plot(duff_arr[:, 0])
plt.title('Duffing x(t)')
plt.grid(True)
plt.show()

# FORECASTING

# for comparison
T_extra = 10000
t_fut = np.linspace(T*dt, (T + T_extra)*dt, T_extra)
forecast = []

cur_z = lz[-1]
duff_state = duff_res[-1].copy() # the last reservoir state
past_states = duff_res[-(m + 1):].copy() # considering last m + 1 duffing res states
past_states = [np.array(i) for i in past_states]

for i in range(T_extra):
    # encoding the last predited z(t) state into f
    cur_z_clipped = np.clip(cur_z, lz_min, lz_max)
    u_norm = (cur_z_clipped - lz_min)/(lz_max - lz_min)
    f = f_min + (f_max - f_min) * u_norm
    f = np.clip(f, f_min, f_max)

    duff_state = rk4(duffing, t[-1] + i*dt, duff_state, dt, f)

    # updating memory
    past_states.append(duff_state.copy())
    if len(past_states) > m + 1:
        past_states.pop(0) # so that it always contains the latest m + 1 states

    stack = [w[j] * past_states[j] for j in range(m + 1)] # taking each of the m + 1 duffing states, multiplying by fading weights as usual
    S_i = np.concatenate(stack) # flattening [x, xdots] into 1D vector
    S_aug = np.concatenate([S_i, S_i**2, S_i**3])
    S_aug /= np.linalg.norm(S_aug) + 1e-6

    next_z = (weights @ S_aug).flatten()[0] # weights @ S_aug is a 1 x 1 matrix which we convert into a 1D vector and we pull out the scalar value
    forecast.append(next_z)
    cur_z = next_z

forecast = np.array(forecast)
forecast_real = forecast * lz_std + lz_mean

plt.plot(t_fut, forecast_real, label = 'Forecasted z(t)')
# plt.plot(t_fut, lz_fut, label = 'Truth Values')
plt.title('Forecasted Lorenz z(t) using Duffing RC')
plt.xlabel('t')
plt.ylabel('z(t)')
plt.grid(True)
plt.legend()
plt.show()