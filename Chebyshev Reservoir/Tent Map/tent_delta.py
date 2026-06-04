import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import pandas as pd
import csv

# Fonts
csfont = {'fontname':'Palatino Linotype', 'size': 15}
hfont = {'fontname':'Palatino Linotype', 'size': 13}

x_min, x_max = -20, 20

def scaling(x, x_min, x_max):
    return (x - x_min)/(x_max - x_min)

def tentmap(r, x):
    if x < 0.5:
        return r*x
    else:
        return r*(1 - x)
    
def tent(x, M, r = 2.0):
    N = len(x)
    T = np.zeros((M + 1, N))
    
    for i, xvalues in enumerate(x):
        state = xvalues
        T[0, i] = state
        
        for n in range(1, M + 1):
            state = tentmap(r, state)
            T[n, i] = state
    
    return T

def lorenz(t, state, sigma = 10, rho = 28, beta = 8/3):
    xt = sigma * (state[1] - state[0])
    yt = state[0] * (rho - state[2]) - state[1]
    zt = state[0] * state[1] - state[2] * beta
    return [xt, yt, zt]

def rc(m, M, alpha, inp, out):
    # simulating the lorenz systen and generating states
    dt = 0.01 # time step for euler
    T_steps = 10000 # total time steps
    time = np.linspace(0, T_steps * dt, T_steps) # timestamps for all steps
    state = ([25.0, 18.0, 120.0])
    l_states = []

    for i in range(T_steps):
        # state += dt * np.array(lorenz(time[i], state))
        # l_states.append(state)
        # explicitly unpacking
        dx, dy, dz = lorenz(time[i], state)
        state += dt * np.array([dx, dy, dz])
        l_states.append(state.copy()) # just using state gave same values of lz

    l_states = np.array(l_states) # each row is one time step and each column is an x, y, z value

    lx = l_states[:, inp]
    # print('input:', len(lx))
    lz = l_states[:, out]
    # print('corr values:', len(lz))
    # print('lz:', lz)

    lx_scaled = scaling(lx, x_min, x_max)
    T = tent(lx_scaled, M, r = 2.0)
    M_1, N = T.shape

    m_weights = alpha ** np.arange(m, -1, -1)

    valid_ts = N -m + 1
    Phi = np.zeros((m * (M_1), valid_ts)) 

    col = 0
    for i in range(m - 1, N):
        feat_vec = []
        for l in range(m):
            t_lag = i - (m - 1 - l)
            feats = T[:, t_lag] # features at this time index for this lag l
            weighted_feats = m_weights[l] * feats

            feat_vec.extend(weighted_feats)

        Phi[:, col] = feat_vec
        col += 1

    # print('Phi:', Phi.shape)

    _lambda = 1e-8

    # allocating some part of data for training and testing
    train_i = int(40/dt)
    train_f = int(49.99/dt)
    pred_i = int(50/dt) # values in the paper

    # adjusting indices for memory as first valid pred is at m- 1
    train_i_corr = train_i - (m - 1)
    train_f_corr = train_f - (m - 1)
    pred_i_corr = pred_i - (m - 1)

    Phi_train = Phi[:, train_i_corr:train_f_corr]
    v_train = lz[train_i:train_f]

    weight_matrix = np.linalg.solve(Phi_train @ Phi_train.T + _lambda * np.eye(m * (M + 1)), Phi_train @ v_train)
    # weight_matrix = v_train @ np.linalg.pinv(Phi_train)

    # predicting on new data
    Phi_test = Phi[:, pred_i_corr:]
    v_test = lz[pred_i:]
    t_test = time[pred_i:]

    pred = weight_matrix @ Phi_test
    delta_norm = np.sqrt(np.mean(((pred - v_test)/v_test)**2))
    log_del = -np.log10(delta_norm) if delta_norm > 0 else np.inf
    return pred, v_test, t_test, log_del, delta_norm

# z from x
m_values = range(20, 500)
delta_m = []
alpha = 0.9
for i in m_values:
    pred, v_test, t_test, log_del, delta = rc(i, 4, alpha, 0, 2)
    delta_m.append(delta)

df_m = pd.DataFrame({'m': list(m_values), 'delta': delta_m})
df_m.to_csv('tnt_del_vs_m_zx.csv', index=False)

a_values = list(np.arange(0.5, 1.0, 0.01))
M = 4
m = 100

delta_a = []

for i in a_values:
    pred, v_test, t_test, log_del, delta = rc(m, M, i, 0, 2)
    delta_a.append(delta)

df_a = pd.DataFrame({'alpha': a_values, 'delta': delta_a})
df_a.to_csv('tnt_del_vs_a_zx.csv', index = False)

# x from z
m_values = range(20, 500)
delta_m = []
for i in m_values:
    pred, v_test, t_test, log_del, delta = rc(i, 4, alpha, 2, 0)
    delta_m.append(delta)

df_m = pd.DataFrame({'m': list(m_values), 'delta': delta_m})
df_m.to_csv('tnt_del_vs_m_xz.csv', index=False)

a_values = list(np.arange(0.5, 1.0, 0.01))
M = 4
m = 100

delta_a = []

for i in a_values:
    pred, v_test, t_test, log_del, delta = rc(m, M, i, 2, 0)
    delta_a.append(delta)

df_a = pd.DataFrame({'alpha': a_values, 'delta': delta_a})
df_a.to_csv('tnt_del_vs_a_xz.csv', index=False)

# y from x
m_values = range(20, 500)
delta_m = []
for i in m_values:
    pred, v_test, t_test, log_del, delta = rc(i, 3, 0.8, 0, 1)
    delta_m.append(delta)

df_m = pd.DataFrame({'m': list(m_values), 'delta': delta_m})
df_m.to_csv('tnt_del_vs_m_yx.csv', index = False)

a_values = list(np.arange(0.5, 1.0, 0.01))
M = 3
m = 100

delta_a = []

for i in a_values:
    pred, v_test, t_test, log_del, delta = rc(m, M, i, 0, 1)
    delta_a.append(delta)

df_a = pd.DataFrame({'alpha': a_values, 'delta': delta_a})
df_a.to_csv('tnt_del_vs_a_yx.csv', index=False)

# x from y
m_values = range(20, 500)
delta_m = []
for i in m_values:
    pred, v_test, t_test, log_del, delta = rc(i, 4, alpha, 1, 0)
    delta_m.append(delta)

df_m = pd.DataFrame({'m': list(m_values), 'delta': delta_m})
df_m.to_csv('tnt_del_vs_m_xy.csv', index=False)

a_values = list(np.arange(0.5, 1.0, 0.01))
M = 4
m = 100

delta_a = []

for i in a_values:
    pred, v_test, t_test, log_del, delta = rc(m, M, i, 1, 0)
    delta_a.append(delta)

df_a = pd.DataFrame({'alpha': a_values, 'delta': delta_a})
df_a.to_csv('tnt_del_vs_a_xy.csv', index=False)