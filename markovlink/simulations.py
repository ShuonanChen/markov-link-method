# simulation utilities 
import numpy as np
import pandas as pd




def random_split(total, k, rng=np.random.default_rng(42)):
    w = rng.dirichlet(np.ones(k))
    return total * w

def renormalize(P_conditional, axis=0):
    P_norm = P_conditional / np.sum(P_conditional, axis=axis, keepdims=True)    
    sums = np.sum(P_norm, axis=axis)
    if not np.allclose(sums, 1.0, rtol=1e-10):
        print(f"Warning: Normalization check failed. Sums: {sums}")    
    return P_norm


def project_rows_to_simplex(A, eps=1e-12):
    # Duchi et al. (ICML 2008) projection onto {p >= 0, sum p = 1}
    A = A.astype(float, copy=False)
    n, m = A.shape
    u = np.sort(A, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1
    ind = np.arange(1, m + 1)
    cond = u - cssv / ind > 0
    rho = cond.sum(axis=1) - 1
    theta = cssv[np.arange(n), rho] / (rho + 1)
    P = np.maximum(A - theta[:, None], 0.0)
    # tiny epsilon to avoid exact zeros if you prefer
    if eps > 0:
        P = np.maximum(P, eps)
        P /= P.sum(axis=1, keepdims=True)
    return P

#   __ _  ___ _ __   ___ _ __ __ _| |_ ___  _ __ 
#  / _` |/ _ \ '_ \ / _ \ '__/ _` | __/ _ \| '__|
# | (_| |  __/ | | |  __/ | | (_| | || (_) | |   
#  \__, |\___|_| |_|\___|_|  \__,_|\__\___/|_|   
#  |___/                                         
# ▓  MAIN FUNCTION: GENERATES P(Y|X,L) WITH CONTROLLED MARKOV ASSUMPTION VIOLATIONS         

def generate_violations(PY_X, Delta_values, num_L, rng, delta_l_distribution ='uniform'):
    num_X, num_Y = PY_X.shape
    results = {}
    for DELTA in Delta_values:
        if delta_l_distribution == 'uniform':
            deltas_L = np.repeat(DELTA/num_L, num_L) # uniform across L 
        elif delta_l_distribution == 'dirichlet':
            deltas_L = rng.dirichlet(np.ones(num_L)) * DELTA  # length of num_L (for each l)                
        PY_XL = np.zeros((num_X, num_Y, num_L))
        for l, d_l in enumerate(deltas_L): # for each L = l
            Z = rng.standard_normal((num_X, num_Y))  # for each PY(Y|X, L=l)
            Z -= Z.mean(axis=1, keepdims=True)  # each row (X) mean = 0
            Z /= np.maximum(np.sum(np.abs(Z), axis=1, keepdims=True), 1e-12)  # after this np.sum(np.abs(Z),-1)
            P = PY_X + d_l * Z
            PY_XL[:, :, l] = project_rows_to_simplex(P) 
#             PY_XL[:, :, l] = P  # miught cause eerror later!
        results[DELTA] = PY_XL
    return results


#   _____   ____ _| |_   _  __ _| |_(_) ___  _ __  
#  / _ \ \ / / _` | | | | |/ _` | __| |/ _ \| '_ \ 
# |  __/\ V / (_| | | |_| | (_| | |_| | (_) | | | |
#  \___| \_/ \__,_|_|\__,_|\__,_|\__|_|\___/|_| |_|
#  METRICS FOR EVALUATING MLM ESTIMATION ACCURACY                                      
#  Computes various distance measures between true and estimated P(Y|X)                

def evaluate_estimation_accuracy(PY_X_true, PY_X_hat, method_name="MLM"):
    num_X = PY_X_true.shape[0]  # Get num_X from the shape of the input
    tv_distance = np.abs(PY_X_true-PY_X_hat).sum()/(2*num_X)
    frobenius_dist = np.linalg.norm(PY_X_true - PY_X_hat, 'fro')
    max_error = np.max(np.abs(PY_X_true - PY_X_hat))
    mse = np.mean((PY_X_true - PY_X_hat)**2)
    return {
        'method': method_name,
        'tv_distance': tv_distance,
        'frobenius': frobenius_dist, 
        'max_error': max_error,
        'mse': mse}


def get_performance_stats(all_simulation_results, Delta_values): # calculate mean, std, CI from these simulations
    df_all_sims = pd.DataFrame(all_simulation_results,)
    summary_stats = []
    for delta in Delta_values:
        delta_results = df_all_sims[df_all_sims['delta'] == delta]    
        summary = {'delta': delta,
            'n_simulations': len(delta_results)}

        for metric in ['tv_distance', 'frobenius', 'max_error', 'mse']:
            values = delta_results[metric].values  #length num_simu
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  
            sem_val = std_val / np.sqrt(len(values))  # Standard error of mean?

            ci_lower = mean_val - 1.96 * sem_val
            ci_upper = mean_val + 1.96 * sem_val 
            summary[f'{metric}_mean'] = mean_val
            summary[f'{metric}_std'] = std_val
            summary[f'{metric}_sem'] = sem_val
            summary[f'{metric}_ci_lower'] = ci_lower
            summary[f'{metric}_ci_upper'] = ci_upper    
        summary_stats.append(summary)
    df_summary = pd.DataFrame(summary_stats)
    return(df_summary)


#  _            _   _              
# | |_ ___  ___| |_(_)_ __   __ _  
# | __/ _ \/ __| __| | '_ \ / _` | 
# | ||  __/\__ \ |_| | | | | (_| | 
#  \__\___||___/\__|_|_| |_|\__, | 
#                           |___/        
#  These functions test the actual DELTA between P(Y|X,L) and P(Y|X)                       
#  Two different approaches for handling probability constraints:                          
#  1. PROJECT: Uses simplex projection (mathematically rigorous)                           
#   2. CLIP: Uses clipping and renormalization (faster but less principled)                  

def generate_violations_project(PY_X, delta, rng):
    num_X, num_Y = PY_X.shape
    Z = rng.standard_normal((num_X, num_Y))
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.maximum(np.sum(np.abs(Z), axis=1, keepdims=True), 1e-12)
    P = PY_X + delta * Z    
    P = project_rows_to_simplex(P)
    return P

def generate_violations_clip(PY_X, delta, rng, eps=1e-12):
    num_X, num_Y = PY_X.shape
    Z = rng.standard_normal((num_X, num_Y))
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.maximum(np.sum(np.abs(Z), axis=1, keepdims=True), 1e-12)
    
    P = PY_X + delta * Z    
    P = np.maximum(P, 0)
    P = np.maximum(P, eps)
    P = P / P.sum(axis=1, keepdims=True)
    return P