import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import pandas as pd
import csv
import time
from datetime import datetime

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

def rc(m, M, alpha, inp, out):
    # simulating the lorenz systen and generating states
    dt = 0.01 # time step for euler
    T_steps = 10000 # total time steps
    time = np.linspace(0, T_steps * dt, T_steps) # timestamps for all steps
    state = ([25.0, 18.0, 120.0])
    l_states = []

    for i in range(T_steps):
        dx, dy, dz = lorenz(time[i], state)
        state += dt * np.array([dx, dy, dz])
        l_states.append(state.copy())

    l_states = np.array(l_states)

    lx = l_states[:, inp]
    lz = l_states[:, out]

    lx_scaled = scaling(lx, x_min, x_max)

    T = chebyshev(lx_scaled, M)
    M_1, N = T.shape

    m_weights = alpha ** np.arange(m, -1, -1)

    valid_ts = N - m + 1
    Phi = np.zeros((m * (M_1), valid_ts)) 

    col = 0
    for i in range(m - 1, N):
        feat_vec = []
        for l in range(m):
            t_lag = i - (m - 1 - l)
            feats = T[:, t_lag]
            weighted_feats = m_weights[l] * feats
            feat_vec.extend(weighted_feats)

        Phi[:, col] = feat_vec
        col += 1

    _lambda = 1e-8

    # allocating some part of data for training and testing
    train_i = int(40/dt)
    train_f = int(49.99/dt)
    pred_i = int(50/dt)

    # adjusting indices for memory as first valid pred is at m- 1
    train_i_corr = train_i - (m - 1)
    train_f_corr = train_f - (m - 1)
    pred_i_corr = pred_i - (m - 1)

    Phi_train = Phi[:, train_i_corr:train_f_corr]
    v_train = lz[train_i:train_f]

    weight_matrix = v_train @ np.linalg.pinv(Phi_train)

    # predicting on new data
    Phi_test = Phi[:, pred_i_corr:]
    v_test = lz[pred_i:]
    t_test = time[pred_i:]

    pred = weight_matrix @ Phi_test
    delta_norm = np.sqrt(np.mean(((pred - v_test)/v_test)**2))
    log_del = -np.log10(delta_norm) if delta_norm > 0 else np.inf
    
    return pred, v_test, t_test, log_del, delta_norm


def master_parameter_sweep(M_values, m_values, alpha_values, inp, out, task_name):
    
    results = []
    total_runs = len(M_values) * len(m_values) * len(alpha_values)
    run_count = 0
    
    print('\n')
    print(f'MASTER SWEEP: {task_name}')
    print('\n')
    print(f'M values: {len(M_values)} ({min(M_values)} to {max(M_values)})')
    print(f'm values: {len(m_values)} ({min(m_values)} to {max(m_values)})')
    print(f'alpha values: {len(alpha_values)} ({min(alpha_values):.2f} to {max(alpha_values):.2f})')
    print(f'Total runs: {total_runs}')
    print(f'Estimated time: {total_runs * 20/3600:.2f} hours (assuming 20s per run)')
    print(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('\n')
    
    start_time = time.time()
    
    for M in M_values:
        for m in m_values:
            for alpha in alpha_values:
                run_count += 1

                pred, v_test, t_test, log_del, delta = rc(m, M, alpha, inp, out)
                    
                results.append({
                    'M': M,
                    'm': m,
                    'alpha': alpha,
                    'delta': delta,
                    'log_delta': log_del
                })
                
                if run_count % 100 == 0 or run_count == total_runs:
                    elapsed = time.time() - start_time
                    rate = run_count/elapsed
                    remaining = (total_runs - run_count)/rate
                    percent = 100 * run_count/total_runs
                    
                    print(f'[{percent:5.1f}%] Run {run_count:5d}/{total_runs} | '
                          f'M = {M:2d}, m = {m:3d}, alpha = {alpha:.2f} | '
                          f'delta = {delta:.6f} | '
                          f'ETA: {remaining/3600:.1f}h')

    results_df = pd.DataFrame(results)
    
    results_df.to_csv(f'master_sweep_{task_name}.csv', index=False)
    
    elapsed_total = time.time() - start_time
    
    print('\n')
    print('SWEEP COMPLETE')
    print('\n')
    print(f'Total time: {elapsed_total/3600:.2f} hours')
    print(f'Results saved to: master_sweep_{task_name}.csv')
    print('\n')
    
    return results_df

M_values = list(range(2, 13)) #--> 11 VALUES
m_values = list(range(50, 501, 20)) #--> 23 VALUES
alpha_values = list(np.arange(0.5, 1.01, 0.05)) #--> 11 VALUES

if __name__ == '__main__':
    tasks = [
        (0, 1, 'y_from_x'),
        (1, 0, 'x_from_y'),
    ]

    for inp, out, task_name in tasks:
        print(f'Starting task: {task_name}\n')
        results_df = master_parameter_sweep(
            M_values = M_values,
            m_values = m_values,
            alpha_values = alpha_values,
            inp = inp,
            out = out,
            task_name = task_name
        )
        
        print(f'Best result: delta = {results_df["delta"].min():.6f}')
        best_idx = results_df['delta'].idxmin()
        best_params = results_df.loc[best_idx]
        print(f'M = {int(best_params["M"])}, m = {int(best_params["m"])}, alpha = {best_params["alpha"]:.3f}')