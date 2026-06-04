import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fonts
csfont = {'fontname':'Palatino Linotype', 'size': 15}
hfont = {'fontname':'Palatino Linotype', 'size': 13}

def logistic(r, x):
    return r*x*(1 - x)

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

state = np.array([25.0, 18.0, 120.0]) # initial conditions
l_states = [] # all states in the lorenz system as the system evolves -- our time series

# the actual application of euler
for i in range(T):
    dx, dy, dz = lorenz(t[i], state) # calculating our derivatives for euler method
    state += dt * np.array([dx, dy, dz])
    l_states.append(state.copy())

l_states = np.array(l_states) # each row is one time step and each column is an x, y, z value

lx = l_states[:, 0]
ly = l_states[:, 1]
lx_raw = lx.copy() # keeping the raw, unnormalized x values for the final plot
ly_raw = ly.copy() # keeping the raw z values for input encoding

# noise
delta = 0.01
noise = np.random.uniform(-delta, delta, size = ly_raw.shape)
ly_raw_noisy = ly_raw + noise

# to run fixed memory prediction directly, but have added as a function below so that it can be run multiple times
u_min, u_max = -17.0, 17.0
'''
r_min, r_max = 3.5, 3.8

ly_min, ly_max = np.min(ly), np.max(ly)
r_enc = r_min + (r_max - r_min) * ((ly_raw - u_min)/(u_max - u_min))

P = 3 # virtual nodes
X = [] # logistic reservoir state
for r in r_enc:
    x = 0.95 # initial value for logistic map from the paper
    trajectory = []
    for i in range(P):
        x = logistic(r, x)
        trajectory.append(x)
    X.append(np.array(trajectory))

# memory stacking as it is a temporal task
m = 100 # increased memory as compared to predicting z and rmse was bad
w = np.linspace(0, 1, m + 1) # fading memory -- weights chosen from a uniform distribution
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
X_aug = X_i

train_start_idx = int(40/dt)
train_end_idx = int(50/dt)
pred_start_idx = int(50/dt)

# X_aug starts at index m, so adjusting indices for memory stacking
train_start_idx_mem = train_start_idx - m
train_end_idx_mem = train_end_idx - m
pred_start_idx_mem = pred_start_idx - m

X_train = X_aug[:, train_start_idx_mem:train_end_idx_mem]
v_train = lx_raw[train_start_idx:train_end_idx].reshape(1, -1)

# training
weights = v_train @ np.linalg.pinv(X_train)

# prediction
X_pred = X_aug[:, pred_start_idx_mem:]
prediction = weights @ X_pred
prediction_real = prediction

# rmse calc
rmse = np.sqrt(np.mean((prediction_real.flatten() - lx_raw[pred_start_idx:])**2))
'''

lambda_max = 0.906 # maximal lyapunov exponent of Lorenz

def run_logistic_rc(P_val, m_val, r_min_val, r_max_val, lx_raw, ly_raw_noisy, u_min, u_max, T_steps, dt):
    r_enc = r_min_val + (r_max_val - r_min_val) * ((ly_raw_noisy - u_min)/(u_max - u_min))
    X = []
    for r in r_enc:
        x = 0.95 # initial value for logistic map from the paper
        trajectory = []
        for i in range(P_val):
            x = logistic(r, x)
            trajectory.append(x)
        X.append(np.array(trajectory))

    # memory stacking as it is a temporal task
    w = np.linspace(0, 1, m_val + 1) # fading memory -- weights chosen from a uniform distribution
    X_i = []
    
    # m_val should be less than T_steps
    if m_val >= T_steps:
        # in this case, we use the maximum possible memory length.
        # m_val max = 1200, T_steps = 10000
        m_val = T_steps - 1 

    for i in range(m_val, T_steps):
        stack = []
        for j in range(m_val + 1):
            past = X[i - m_val + j]
            weighted = w[j] * past
            stack.append(weighted)
        stack = np.concatenate(stack) # concatenate instead of np.array(stack)
        X_i.append(stack)

    X_i = np.array(X_i).T
    X_aug = X_i

    train_start_idx = int(40/dt)
    train_end_idx = int(50/dt)
    pred_start_idx = int(50/dt)

    # X_aug starts at index m, so adjusting indices for memory stacking
    train_start_idx_mem = train_start_idx - m_val
    train_end_idx_mem = train_end_idx - m_val
    pred_start_idx_mem = pred_start_idx - m_val

    X_train = X_aug[:, train_start_idx_mem:train_end_idx_mem]
    v_train = lx_raw[train_start_idx:train_end_idx].reshape(1, -1)

    # training
    weights = v_train @ np.linalg.pinv(X_train)

    # prediction
    X_pred = X_aug[:, pred_start_idx_mem:]
    prediction = weights @ X_pred
    prediction_real = prediction

    # rmse calc
    rmse = np.sqrt(np.mean((prediction_real.flatten() - lx_raw[pred_start_idx:])**2))

    '''
    # using the last 1000 prediction steps (10s)
    train_duration_steps = int(10/dt)
    pred_duration_steps = int(10/dt)

    pred_end_idx = T_steps
    pred_start_idx = T_steps - pred_duration_steps

    train_end_idx = pred_start_idx
    train_start_idx = pred_start_idx - train_duration_steps
    
    # X_aug starts at index m_val, so adjusting indices for memory stacking
    train_start_idx_mem = train_start_idx - m_val
    train_end_idx_mem = train_end_idx - m_val
    pred_start_idx_mem = pred_start_idx - m_val
    pred_end_idx_mem = pred_end_idx - m_val

    X_train = X_aug[:, train_start_idx_mem:train_end_idx_mem]
    v_train = lx_raw[train_start_idx:train_end_idx].reshape(1, -1)

    # training
    weights = v_train @ np.linalg.pinv(X_train, rcond = 1e-5)

    # prediction
    X_pred = X_aug[:, pred_start_idx_mem:pred_end_idx_mem]
    prediction = weights @ X_pred
    prediction_real = prediction

    rmse = np.sqrt(np.mean((prediction_real.flatten() - lx_raw[pred_start_idx:pred_end_idx])**2))
    '''
    
    return rmse

# continuation of plotting the prediction for fixed memory
'''
fig, ax = plt.subplots(1, 1)

ax.plot(t[pred_start_idx:], lx_raw[pred_start_idx:], label = 'Truth Values')
ax.plot(t[pred_start_idx:], prediction_real.flatten(), '--', label = f'Predicted Values; RMSE = {round(rmse, 4)}')

# to display in terms of maximal lypunov exp
def t_to_tl(t_val):
    return t_val * lambda_max

def tl_to_t(tl_val):
    return tl_val/lambda_max

# secondary x axis (t_L)
secax = ax.secondary_xaxis('top', functions=(t_to_tl, tl_to_t))
secax.set_xlabel('$t_{L} = t \cdot \lambda_{max}$ ($\lambda_{max} =  0.906$)')
secax.tick_params(axis = 'x', direction= 'in', top= True)

ax.set_title(f'Logistic RC to Predict Lorenz x(t) from y(t); $\\delta = {delta}$; Memory: {m}; [3.5, 3.8]')
ax.set_xlabel('t')
ax.set_ylabel('y(t)')
ax.legend()
plt.show()
'''

plt.figure(figsize = (24, 6)) 
plt.subplots_adjust(wspace = 0.3)

# plot (a)
P_a = 3
m_a = 1000
r_max_vals_a = np.linspace(1.1, 4.0, 30) 
r_min_ranges_a = [
    {'min': 1.5, 'color': 'midnightblue', 'label': r'$a_{\min} = 1.5$'},
    {'min': 3.2, 'color': 'plum', 'label': r'$r_{\min} = 3.2$'},
    {'min': 3.5, 'color': 'black', 'label': r'$r_{\min} = 3.5$'},
    {'min': 3.6, 'color': 'teal', 'label': r'$r_{\min} = 3.6$'}
]

plt.subplot(1, 3, 1)

all_data_a = []
for r_spec in r_min_ranges_a:
    r_min_a = r_spec['min']
    rmse_vals_a = []
    
    for r_max_a in r_max_vals_a:
        try:
            if r_max_a >= r_min_a:
                rmse_a = run_logistic_rc(P_a, m_a, r_min_a, r_max_a, lx_raw, ly_raw_noisy, u_min, u_max, T, dt)
                rmse_vals_a.append(rmse_a)
            else:
                # NaN for invalid ranges where a_max < a_min
                rmse_vals_a.append(np.nan) 
        except:
            rmse_vals_a.append(np.nan)
    
    plt.plot(r_max_vals_a, rmse_vals_a, 'o-', color = r_spec['color'], label = r_spec['label'], markersize = 4)
    all_data_a.append({'r_min': r_min_a, 'r_max': r_max_vals_a, 'RMSE': rmse_vals_a})

df_combined = pd.DataFrame()
for data in all_data_a:
    df_temp = pd.DataFrame({'a_min': data['r_min'], 'a_max': data['r_max'], 'RMSE': data['RMSE']})
    df_combined = pd.concat([df_combined, df_temp], ignore_index = True)
df_combined.to_csv('output1.csv', index = False)

plt.title(r'RMSE vs $r_{max}$ ($M = 3, m = 1000, r_{min} = 1$)', **csfont)
plt.xlabel('$r_{max}$', **hfont)
plt.ylabel('RMSE', **hfont)
plt.legend()

# plot (b)
m_b = 1000
P_vals_b = np.arange(1, 11) # P from 1 to 10
ranges_b = [
    {'min': 1.0, 'max': 2.0, 'label': r'$[1.0, 2.0]$ (stable)', 'color': 'teal'},
    {'min': 3.57, 'max': 3.58, 'label': r'$[3.57, 3.58]$ (edge)', 'color': 'midnightblue'},
    {'min': 3.8, 'max': 3.9, 'label': r'$[3.8, 3.9]$ (chaotic)', 'color': 'plum'}
]

plt.subplot(1, 3, 2) 

for r in ranges_b:
    rmse_vals_b = []
    for P_b in P_vals_b:
        try:
            rmse_b = run_logistic_rc(P_b, m_b, r['min'], r['max'], lx_raw, ly_raw_noisy, u_min, u_max, T, dt)
            rmse_vals_b.append(-np.log10(rmse_b))
        except:
            rmse_vals_b.append(np.nan)

    plt.plot(P_vals_b, rmse_vals_b, 'o-', color = r['color'], label = r['label'])

plt.title(r'-$\log_{10}(\mathrm{RMSE})$ vs $M$ ($m = 1000$)', **csfont)
plt.xlabel(r'$M$', **hfont)
plt.ylabel(r'-$\log_{10}(\mathrm{RMSE})$', **hfont)
plt.xticks(P_vals_b)
plt.legend()

# plot (c)
P_c = 3
r_min_c = 1.0
r_max_c = 2.0
m_vals_c = np.arange(100, 1201, 50) # m from 100 to 1200
rmse_vals_c = []

for m_c in m_vals_c:
    try:
        rmse_c = run_logistic_rc(P_c, m_c, r_min_c, r_max_c, lx_raw, ly_raw_noisy, u_min, u_max, T, dt)
        rmse_vals_c.append(-np.log10(rmse_c))
    except:
        rmse_vals_c.append(np.nan)

data = {'m': m_vals_c, 'RMSE': rmse_vals_c}

df = pd.DataFrame(data)
df.to_csv('output3.csv', index = False)

plt.subplot(1, 3, 3)
plt.plot(m_vals_c, rmse_vals_c, 'o-', color = 'midnightblue', label = r'$M = 3, r \in [1, 2]$')
plt.title(r'$-\log_{10}(\mathrm{RMSE})$ vs $m$ ($M = 3, r \in [1, 2]$)', **csfont)
plt.xlabel('$m$', **hfont)
plt.ylabel(r'$-\log_{10}(\mathrm{RMSE})$', **hfont)
plt.legend()
plt.tight_layout()
plt.savefig('fig12_1.png')