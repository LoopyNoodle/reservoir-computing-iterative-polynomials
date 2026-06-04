import numpy as np
import matplotlib.pyplot as plt
import csv

def logistic(r, x):
    return r * x * (1 - x)

def lorenz(t, state, sigma = 10, rho = 28, beta = 8/3):
    xt = sigma * (state[1] - state[0])
    yt = state[0] * (rho - state[2]) - state[1]
    zt = state[0] * state[1] - state[2] * beta
    return [xt, yt, zt]

def rc(m):
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

    '''
    for i in range(T_steps):
        t_i = time[i]
        k1 = dt * np.array(lorenz(t_i, state))
        k2  = dt * np.array(lorenz(t_i + dt/2, state + k1/2))
        k3 = dt * np.array(lorenz(t_i + dt/2, state + k2/2))
        k4 = dt * np.array(lorenz(t_i + dt, state + k3))

        state = state + (k1 + 2 * k2 + 2 * k3 + k4)/6
        l_states.append(state)

    l_states = np.array(l_states)
    '''

    lx = l_states[:, 1]
    # print('input:', len(lx))
    lz = l_states[:, 0]
    # print('corr values:', len(lz))
    # print('lz:', lz)

    # adding noise
    delta = 0.00
    noise = np.random.uniform(-delta, delta, size = lz.shape)
    lx_noisy = lx + noise
    # print('corr values w noise:', len(lx_noisy))

    # logistic map input encoding
    transients, N_steps = 100, 250
    r_min, r_max = 3.5, 4.0
    # lx_max, lx_min = max(lx_noisy), min(lx_noisy)
    lx_min, lx_max = -17.0, 17.0 # used in Arun et al.

    r_enc = r_min + (r_max - r_min) * ((lx_noisy - lx_min)/(lx_max - lx_min))
    # print('r_enc:', len(r_enc))
    # print('values of r_enc:', r_enc)

    P = 3
    X = [] # logistic reservoir state
    for r in r_enc:
        x = 0.95 # initial state
        traj = []
        for i in range(P):
            x = logistic(r, x)
            traj.append(x)
        X.append(np.array(traj))

        '''the below algorithm was what was originally considered. However, in Arun et. al, they
        ran the logistic map for P (virtual nodes) = 3 times'''
        # for i in range(N_steps):
        #     x = logistic(r, x)
        #     if i >= transients:
        #         traj.append(x)
        # for i in range (0, len(traj)):
        #     if np.isnan(traj[i]) == True or np.isinf(traj[i]) == True:
        #             traj[i] = 0
        # X.append(np.array(traj))
        '''this takes too much RAM and system crashes^^'''
        # for i in range(N_steps):
        #     x = logistic(r, x)
        #     if i >= transients:
        #         traj.append(x)
        # traj = np.array(traj)
        # for i in range (0, len(traj)):
        #     if np.isnan(traj[i]) == True or np.isinf(traj[i]) == True:
        #             traj[i] = 0
        '''for 'SVD did not converge' error from np.linalg.pinv(), 
        saw NaN vals in X_i'''
        # feats = [np.mean(traj), np.std(traj), np.max(traj), np.min(traj)]
        # X.append(np.array(feats))
        '''this is more efficient. Each X_i becomes a 4D vector which contains a 
        summary of statistics and still contains info about the dynamics and chaotic means hight std'''

    # print('X:', np.array(X).shape)
    # print('X values:', X)

    # memory stacking as it is a temporal task
    # increase memory > 100 for predicting x as rmse was bad :/
    m_weights = np.linspace(0, 1, m + 1) # uniform dist weights

    X_i = []
    for i in range(m, T_steps):
        m_stack = []
        for j in range(0, m + 1):
            val = X[i - m + j]
            weighted = m_weights[j] * val
            m_stack.append(weighted)
        m_stack = np.concatenate(m_stack) # concatenate instead of np.array(stack)
        X_i.append(m_stack)

    X_i = np.array(X_i).T
    # print('X_i:', X_i.shape)
    # print('X_i values:', X_i)

    # allocating some part of data for training and testing
    train_i = int(40/dt)
    train_f = int(49.99/dt)
    pred_i = int(50/dt) # values in the paper

    # adjusting indices for memory stacking
    train_i_corr = train_i - m
    train_f_corr = train_f - m
    pred_i_corr = pred_i - m

    X_train = X_i[:, train_i_corr : train_f_corr]
    # print('X_train:', X_train.shape)
    # print('X_train values:', X_train)
    v_train = lz[train_i : train_f].reshape(1, -1)
    # print('v_train:', v_train.shape)
    # print('v_train values:', v_train)
    # print('pseudoinverse:', (np.linalg.pinv(X_train)).shape)

    weight_matrix = v_train @ np.linalg.pinv(X_train)
    # print('weight_matrix:', weight_matrix.shape)
    # print('wm values:', weight_matrix)

    # prediction and rmse
    X_pred = X_i[:, pred_i_corr:]
    # print('X_pred:', X_pred.shape)
    pred = weight_matrix @ X_pred
    # print('pred:', pred.shape)
    # print('pred values:', pred)

    delta = np.mean((pred.flatten() - lz[pred_i:])**2)

    return delta

# rmse, delta = rc(100)
m_vals = np.arange(100, 1200, 5)
deltas = []

for i, m in enumerate(m_vals):
    delta = rc(m)
    deltas.append(delta)
    if (i + 1) % 10 == 0:
        print(f'Progress: {i+1}/{len(m_vals)} runs completed (m = {m})')

# Save to CSV
with open('logistic_error_vs_memory_xy.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['m', 'delta'])
    for m, delta in zip(m_vals, deltas):
        writer.writerow([m, delta])