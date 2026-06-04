import numpy as np
import matplotlib.pyplot as plt
import csv

x_min, x_max = -20, 20

def scaling(x, x_min, x_max):
    # scale input from [x_min, x_max] to [-1, 1]
    return 2 * (x - x_min)/(x_max - x_min) - 1

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

def lorenz(t, state, sigma = 10, rho = 28, beta = 8/3):
    xt = sigma * (state[1] - state[0])
    yt = state[0] * (rho - state[2]) - state[1]
    zt = state[0] * state[1] - state[2] * beta
    return [xt, yt, zt]

def rc(m, M):
    # simulating the lorenz systen and generating states
    dt = 0.01 # time step for euler
    T_steps = 10000 # total time steps
    time = np.linspace(0, T_steps * dt, T_steps) # timestamps for all steps
    state = ([25.0, 18.0, 120.0])
    l_states = []

    for i in range(T_steps):
        # state += dt * np.array(lorenz(time[i], state))
        # l_states.append(state)
        # explicitly unpack
        dx, dy, dz = lorenz(time[i], state)
        state += dt * np.array([dx, dy, dz])
        l_states.append(state.copy()) # just using state gave same values of lz

    l_states = np.array(l_states) # each row is one time step and each column is an x, y, z value

    lx = l_states[:, 0]
    # print('input:', len(lx))
    lz = l_states[:, 2]
    # print('corr values:', len(lz))
    # print('lz:', lz)

    lx_scaled = scaling(lx, x_min, x_max)

    T = chebyshev(lx_scaled, M)
    M_1, N = T.shape

    alpha = 0.9
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

    print('Phi:', Phi.shape)

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

    # weight_matrix = np.linalg.solve(Phi_train @ Phi_train.T + _lambda * np.eye(m * (M + 1)), Phi_train @ v_train)
    weight_matrix = v_train @ np.linalg.pinv(Phi_train)

    # predicting on new data
    Phi_test = Phi[:, pred_i_corr:]
    v_test = lz[pred_i:]
    t_test = time[pred_i:]

    pred = weight_matrix @ Phi_test
    delta_norm = np.mean((pred - v_test)**2)
    log_del = -np.log10(delta_norm) if delta_norm > 0 else np.inf
    return pred, v_test, t_test, log_del, delta_norm

pred, v_test, t_test, log_del, delta = rc(100, 9)

plt.figure()

# plt.subplot(1, 2, 1)
plt.plot(t_test, v_test, '-', linewidth = 2, label ='Reference Values')
plt.plot(t_test, pred, '--', linewidth = 2, label = f'Predicted Values; $\delta$ = {round(delta, 4)}, $-\log(\delta) = {round(log_del, 2)}$')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('z(t)', fontsize=12)
plt.title('Lorenz z(t) from x(t); M = 9, m = 100', fontsize = 14)
plt.legend()
plt.show()