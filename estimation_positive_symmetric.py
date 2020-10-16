"""
This script reproduces the results of section 4.2 on simulated data with positive symmetric coefficients.
It is divided into separate sections.
- Data generation
- SPLRD model (ours): parameter selection
- SPLRD model (ours): evaluation
- SPLR estimation (Richard et al. 2012) on T: parameter selection
- SPLR estimation (Richard et al. 2012) on T: evaluation
- SPLR estimation (Richard et al. 2012) on V: parameter selection
- SPLR estimation (Richard et al. 2012) on V: evaluation
- Graph features computation
The procedures for each model are very similar.

Generally, the naming conventions in the code go as follows:
- [...]_init is the generated uncorrupted data:
    - T_init is the true template
    - Vs_init is the true deviations
    - V_init is the true deviations, stacked in a tall matrix.
- T and V without init refer to the estimates computed through the optimization.
- The "pos" suffix refers to positive data.


REMARK
When running the code on different platforms, we noticed that the code produced
slightly different results, due to differences in the distributions of numpy.
In order to make the code yield identical results on different platforms,
we round off the parameters optimization objective to 6 decimal places.
"""

import numpy as np
import networkx as nx
from tqdm import *
import skopt
import os

from src.utils import *
from src.generate import dataset
import src.douglas_rachford as dr


# =======================================================
# =========== GLOBAL EXPERIMENT PARAMETERS ==============
# =======================================================

log_dir = "output" # Directory for the logs and output
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

n = 100         # number of nodes
n_samples = 10  # number of samples
n_train = 5     # number of test data sets
n_test = 5      # number of test data sets

# Optimization parameters for the Douglas-Rachford algorithm
theta_dr = 0.9
tau_dr = 0.1

niter = 200 # Number of optimization iterations for both algorithms


# ====================================================================
# ================ DATA GENERATION                 ===================
# ================ Positive symmetric coefficients ===================
# ====================================================================

print("\nGenerating the data sets.")
np.random.seed(0)

# Generate train data with positive symmetric coefficients

samples_pos_train = np.zeros((n_train, n_samples, n, n)) # Model samples
T_init_pos_train = np.zeros((n_train, n, n))             # Template matrices
M_pos_train = np.zeros((n_train, n, n))                  # Mean samples
Vs_init_pos_train = np.zeros((n_train, n_samples, n, n)) # Deformations matrices
V_init_pos_train = np.zeros((n_train, n_samples * n, n)) # Stacked deformations

for k in range(n_train):
    samples_pos_train[k], T_init_pos_train[k], Vs_init_pos_train[k], _ = dataset(
                                        n, n_samples,
                                        T_rank=10, T_sparse=0.7, T_positive=True,
                                        V_rank=10, V_sparse=0.7, V_positive=True,
                                        noise_sparsity=0.7, noise_amplitude=1, noise_positive=True, symmetric=True,
                                       )
    V_init_pos_train[k] = np.vstack(Vs_init_pos_train[k])
    M_pos_train[k] = samples_pos_train[k].mean(axis=0)


# Generate test data with positive symmetric coefficients

samples_pos_test = np.zeros((n_test, n_samples, n, n)) # Model samples
T_init_pos_test = np.zeros((n_test, n, n))             # Template matrices
M_pos_test = np.zeros((n_test, n, n))                  # Mean samples
Vs_init_pos_test = np.zeros((n_test, n_samples, n, n)) # Deformations matrices
V_init_pos_test = np.zeros((n_test, n_samples * n, n)) # Stacked deformations

for k in range(n_test):
    samples_pos_test[k], T_init_pos_test[k], Vs_init_pos_test[k], _ = dataset(
                                        n, n_samples,
                                        T_rank=10, T_sparse=0.7, T_positive=True,
                                        V_rank=10, V_sparse=0.7, V_positive=True,
                                        noise_sparsity=0.7, noise_amplitude=1, noise_positive=True, symmetric=True,
                                       )
    V_init_pos_test[k] = np.vstack(Vs_init_pos_test[k])
    M_pos_test[k] = samples_pos_test[k].mean(axis=0)


# Estimation error between M and T
errors_pos_M = np.array([relative_error(M_pos_test[k], T_init_pos_test[k]) for k in range(n_test)])
# Estimation error between A_i-M and V_i
errors_pos_A_M = np.array([relative_error(samples_pos_test[k]-M_pos_test[k][None,:,:], Vs_init_pos_test[k]) for k in range(n_test)])


f = open(f"{log_dir}/mean_pos.txt", "a")
f.writelines([
    f"Average relative RMSE between the mean sample and the template: {np.mean(errors_pos_M)} +/- {np.std(errors_pos_M)}\n",
    f"Average relative RMSE between A_i-M and V_i: {np.mean(errors_pos_A_M)} +/- {np.std(errors_pos_A_M)}\n"
])
f.close()


# ========================================================
# ============== SPLRD MODEL WITH L1 LOSS ================
# ============== Parameter selection      ================
# ========================================================

# We choose the parameters that best reconstruct the deviations V


print("\nL1 model: parameters selection")
np.random.seed(0) # Set the random seed to zero for reproducibility

# Define the optimization space
SPACE = [
    skopt.space.Real(0, 10, name='lambd', prior='uniform'),
    skopt.space.Real(0, 10, name='rho', prior='uniform'),
    skopt.space.Real(0, 10, name='mu', prior='uniform'),
    skopt.space.Real(0, 10, name='nu', prior='uniform'),
]

STATIC_PARAMS = {
    'theta': theta_dr,
    'tau' : tau_dr,
    'niter': niter,
}

cnt = Counter()
progress = tqdm(total=100)

# Objective to be optimized for parameter selection
@skopt.utils.use_named_args(SPACE)
def objective(**params):
    global cnt
    cnt.increment()

    all_params = {**params, **STATIC_PARAMS}
    result = 0

    for k in range(n_train):
        m = dr.SparseLowRank(**all_params) # Define Douglas-Rachford optimizer
        m.fit(samples_pos_train[k])        # Perform the decomposition on each training set
        # We compute the relative error between the estimated V_i and the true V_i
        result += relative_error(m.V, V_init_pos_train[k])

    # Save step results
    f = open(f"{log_dir}/L1_pos.txt", "a")
    f.write(f"===== Training step {cnt.k} =====\n")
    f.write(f"Parameters: "+str(params)+"\n")
    f.write(f"Total training relative RMSE: {result}\n")
    f.close()

    # Update progress bar
    progress.update(1)
    progress.set_postfix({"objective": result})
    return np.round(result, 6)

# Execute parameter optimization
results_dr_pos = skopt.forest_minimize(objective, SPACE, n_calls=100, random_state=0)

f = open(f"{log_dir}/L1_pos.txt", "a")
f.write(f"\n\nBest training relative RMSE: {np.round(results_dr_pos['fun'], 4)}\n")
f.write(f"Best training parameters: "+str(results_dr_pos['x'])+"\n")
f.close()


# ========================================================
# ============== SPLRD MODEL WITH L1 LOSS ================
# ============== Evaluation on test data  ================
# ========================================================

print("\nL1 model: evaluation")

models_dr_pos_test = []

# Estimate the template and deviations on each test data set
for k in trange(n_test):
    models_dr_pos_test.append(dr.SparseLowRank(*results_dr_pos['x'], theta=theta_dr, tau=tau_dr, niter=niter))
    models_dr_pos_test[k].fit(samples_pos_test[k])


# Compute the reconstruction errors
# Between the estimated template and the true template
errors_dr_pos_T   = [relative_error(models_dr_pos_test[k].T, T_init_pos_test[k]) for k in range(n_test)]
# Between the estimated deviation and the true deviation
errors_dr_pos_V   = [relative_error(models_dr_pos_test[k].V, V_init_pos_test[k]) for k in range(n_test)]
# Between (A_i-[estimated template]) and the true deviation
errors_dr_pos_A_T = [relative_error(samples_pos_test[k]-models_dr_pos_test[k].T[None,:,:], Vs_init_pos_test[k]) for k in range(n_test)]


# Save the results
f = open(f"{log_dir}/L1_pos_test.txt", "a")
f.writelines([
    "Best parameters for the L1 model with positive symmetric coefficients:\n",
    f"lambda = {np.round(results_dr_pos['x'][0], 3)}\n",
    f"rho    = {np.round(results_dr_pos['x'][1], 3)}\n"
    f"mu     = {np.round(results_dr_pos['x'][2], 3)}\n"
    f"nu     = {np.round(results_dr_pos['x'][3], 3)}\n \n",
    f"Average relative RMSE for the template  T = {np.mean(errors_dr_pos_T)} +/- {np.std(errors_dr_pos_T)}\n",
    f"Average relative RMSE for the deviations V = {np.mean(errors_dr_pos_V)} +/- {np.std(errors_dr_pos_V)}\n",
    f"Average relative RMSE for the deviations V estimated by A_i-T = {np.mean(errors_dr_pos_A_T)} +/- {np.std(errors_dr_pos_A_T)}\n",
             ])
f.close()


# ========================================================
# ================ SPLR DENOISING FOR T ==================
# ================ Parameter selection  ==================
# ========================================================

# In this section, we compute what we denoted T^M in the paper
# We choose the parameters that best reconstruct the template T

print("\nSPLR T^M: parameters selection")
np.random.seed(0)

# Define the optimization space
SPACE_DENOISE = [
    skopt.space.Real(0, 10, name='lambd', prior='uniform'),
    skopt.space.Real(0, 10, name='rho', prior='uniform'),
]

cnt = Counter()
progress = tqdm(total=100)

# Objective to be optimized for parameter selection
@skopt.utils.use_named_args(SPACE_DENOISE)
def objective_denoise(**params):
    global cnt
    cnt.increment()

    result = 0
    for k in range(n_train):
        TM_pos_train = dr.single_sample_splr(M_pos_train[k], params['lambd'], params['rho'],
                                             theta_dr, tau_dr, niter)
        result += relative_error(TM_pos_train, T_init_pos_train[k])

    # Save step results
    f = open(f"{log_dir}/splr_T_pos.txt", "a")
    f.write(f"===== Training step {cnt.k} =====\n")
    f.write(f"Parameters: "+str(params)+"\n")
    f.write(f"Total training relative RMSE: {result}\n")
    f.close()

    # Update progress bar
    progress.update(1)
    progress.set_postfix({"obj": result})
    return np.round(result, 6)

# Execute the parameter optimization
results_denoise_pos_T = skopt.forest_minimize(objective_denoise, SPACE_DENOISE, n_calls=100, random_state=0)

f = open(f"{log_dir}/splr_T_pos.txt", "a")
f.write(f"\n\nBest training relative RMSE: {np.round(results_denoise_pos_T['fun'], 4)}\n")
f.write(f"Best training parameters: "+str(results_denoise_pos_T['x'])+"\n")
f.close()


# ===========================================================
# ================ SPLR DENOISING FOR T    ==================
# ================ Evaluation on test data ==================
# ===========================================================

print("\nSPLR T^M: evaluation")

TM_pos_test = np.zeros((n_test, n, n))

for k in trange(n_test):
    TM_pos_test[k] = dr.single_sample_splr(M_pos_test[k], *results_denoise_pos_T['x'], theta_dr, tau_dr, niter)

# Compute the reconstruction error for T^M
errors_dr_pos_TM = [relative_error(TM_pos_test[k], T_init_pos_test[k]) for k in range(n_test)]

# Save the result
f = open(f"{log_dir}/splr_T_pos_test.txt", "a")
f.writelines([
    "Best parameters for T^M with positive symmetric coefficients:\n",
    f"lambda = {np.round(results_denoise_pos_T['x'][0], 3)}\n",
    f"rho    = {np.round(results_denoise_pos_T['x'][1], 3)}\n\n"
    f"Average relative RMSE for the template  T = {np.mean(errors_dr_pos_TM)} +/- {np.std(errors_dr_pos_TM)}\n",
             ])
f.close()


# ========================================================
# ================ SPLR DENOISING FOR V ==================
# ================ Parameter selection  ==================
# ========================================================

# In this section, we compute what we denoted V^M in the paper
# We choose the parameters that best reconstruct the template T

print("\nSPLR V^M: parameters selection")

np.random.seed(0)

# Define the optimization space
SPACE_DENOISE = [
    skopt.space.Real(0, 10, name='mu', prior='uniform'),
    skopt.space.Real(0, 10, name='nu', prior='uniform'),
]

# First, we compute T^M with the best parameters obtained in the previous code section
TM_pos_train = np.zeros_like(T_init_pos_train)
for k in range(n_train):
    TM_pos_train[k] = dr.single_sample_splr(M_pos_train[k], *results_denoise_pos_T['x'],
                                        theta_dr, tau_dr, niter)

cnt = Counter()
progress = tqdm(total=100)

# Objective to be optimized for parameter selection
@skopt.utils.use_named_args(SPACE_DENOISE)
def objective_denoise(**params):
    global cnt
    cnt.increment()

    result = 0

    for k in range(n_train):
        V = np.zeros((n*n_samples, n))
        for i in range(n_samples):
            # Perform sparse low rank denoising from Richard et al. (2012) on A_i - T^M
            V[i*n:(i+1)*n] = dr.single_sample_splr(samples_pos_train[k,i]-TM_pos_train[k],
                                                   params['mu'], params['nu'], theta_dr, tau_dr, niter)
        result += relative_error(V, V_init_pos_train[k])

    # Save step result
    f = open(f"{log_dir}/splr_V_pos.txt", "a")
    f.write(f"===== Training step {cnt.k} =====\n")
    f.write(f"Parameters: "+str(params)+"\n")
    f.write(f"Total training relative RMSE: {result}\n")
    f.close()

    # Update progress bar
    progress.update(1)
    progress.set_postfix({"obj": result})
    return np.round(result, 6)

# Execute parameter optimization
results_denoise_pos_V = skopt.forest_minimize(objective_denoise, SPACE_DENOISE, n_calls=100, random_state=0)
f = open(f"{log_dir}/splr_V_pos.txt", "a")
f.write(f"\n\nBest training relative RMSE: {np.round(results_denoise_pos_V['fun'], 4)}\n")
f.write(f"Best training parameters: "+str(results_denoise_pos_V['x'])+"\n")
f.close()


# ===========================================================
# ================ SPLR DENOISING FOR V    ==================
# ================ Evaluation on test data ==================
# ===========================================================

print("\nSPLR V^M: evaluation")

results_denoise_pos_V['x']

VM_pos_test = np.zeros((n_test, n_samples, n, n))

# Perform the decomposition for each V_i in each test data set
for k in trange(n_test):
    for i in range(n_samples):
        VM_pos_test[k,i] = dr.single_sample_splr(samples_pos_test[k,i]-TM_pos_test[k], *results_denoise_pos_V['x'], theta_dr, tau_dr, niter)

# Compute the reconstruction error for V^M
errors_dr_pos_VM = [relative_error(VM_pos_test[k], Vs_init_pos_test[k]) for k in range(n_test)]

# Save the results
f = open(f"{log_dir}/splr_V_pos_test.txt", "a")
f.writelines([
    "Best parameters for V^M with positive symmetric coefficients:\n",
    f"mu    = {np.round(results_denoise_pos_V['x'][0], 3)}\n",
    f"nu    = {np.round(results_denoise_pos_V['x'][1], 3)}\n\n"
    f"Average relative RMSE for the deviations V = {np.mean(errors_dr_pos_VM)} +/- {np.std(errors_dr_pos_VM)}\n",
             ])
f.close()



# =============================================================
# ================ GRAPH FEATURES ESTIMATION ==================
# =============================================================

print("\nComputing graph features.")

np.random.seed(0)

def average_clustering(adj):
    """Weighted clustering coefficient with geometric mean from Opsahl et Panzarasa (2009)"""
    A2 = np.sign(adj)*(np.abs(adj)**(1/2))
    A3 = np.sign(adj)*(np.abs(adj)**(1/3))
    all_triangles = np.einsum('ij, ik -> ijk', A2, A2)
    closed_triangles = np.einsum('ij, ik, jk -> ijk', A3, A3, A3)
    total_triangles = all_triangles.sum(where=(closed_triangles==0)) + closed_triangles.sum()
    return closed_triangles.sum()/total_triangles

def distance(u, v, d):
    """Distance used in the shortest path computation. The 0.1 factor allows to handle the case of unconnected graphs"""
    return 1/(0.1+d["weight"])


# charpath = Characteristic path length = average shortest path length
charpath_true = np.zeros((n_test, n_samples))    # Uncorrupted samples (T+V_i)
charpath_samples = np.zeros((n_test, n_samples)) # Noisy samples
charpath_dr = np.zeros((n_test, n_samples))      # T+V_i reconstruction with the L1 model
charpath_TM_VM = np.zeros((n_test, n_samples))   # T^M+V^M_i

# Weighted clustering coefficient
clustering_true = np.zeros((n_test, n_samples))    # Uncorrupted samples (T+V_i)
clustering_samples = np.zeros((n_test, n_samples)) # Noisy samples
clustering_dr = np.zeros((n_test, n_samples))      # T+V_i reconstruction with the L1 model
clustering_TM_VM = np.zeros((n_test, n_samples))   # T^M+V^M_i

# Weighted average degree
degree_true = np.zeros((n_test, n_samples))    # Uncorrupted samples (T+V_i)
degree_samples = np.zeros((n_test, n_samples)) # Noisy samples
degree_dr = np.zeros((n_test, n_samples))      # T+V_i reconstruction with the L1 model
degree_TM_VM = np.zeros((n_test, n_samples))   # T^M+V^M_i


# Compute all graph features for each test data set
for k in trange(n_test):
    for i in range(n_samples):
        mat = T_init_pos_test[k] + Vs_init_pos_test[k,i]
        G = nx.Graph(mat+1e-5) # Force networkx to create every edge
        charpath_true[k,i] = nx.average_shortest_path_length(G, weight=distance)
        clustering_true[k,i] = average_clustering(mat)
        degree_true[k,i] = mat.mean()

        mat = samples_pos_test[k,i]
        G = nx.Graph(mat+1e-5) # Force networkx to create every edge
        charpath_samples[k,i] = nx.average_shortest_path_length(G, weight=distance)
        clustering_samples[k,i] = average_clustering(mat)
        degree_samples[k,i] = mat.mean()

        mat = models_dr_pos_test[k].S + models_dr_pos_test[k].Ws[i]
        G = nx.Graph(mat+1e-5) # Force networkx to create every edge
        charpath_dr[k,i] = nx.average_shortest_path_length(G, weight=distance)
        clustering_dr[k,i] = average_clustering(mat)
        degree_dr[k,i] = mat.mean()

        mat = TM_pos_test[k] + VM_pos_test[k,i]
        G = nx.Graph(mat+1e-5) # Force networkx to create every edge
        charpath_TM_VM[k,i] = nx.average_shortest_path_length(G, weight=distance)
        clustering_TM_VM[k,i] = average_clustering(mat)
        degree_TM_VM[k,i] = mat.mean()


# Save the results
f = open(f"{log_dir}/graph_params.txt", "a")
f.write(f"Average relative RMSE for charpath_samples: {np.mean(np.abs(charpath_samples-charpath_true)/charpath_true)} +/- {np.std(np.abs(charpath_samples-charpath_true)/charpath_true)}\n")
f.write(f"Average relative RMSE for charpath_dr:      {np.mean(np.abs(charpath_dr-charpath_true)/charpath_true)} +/- {np.std(np.abs(charpath_dr-charpath_true)/charpath_true)}\n")
f.write(f"Average relative RMSE for clustering_samples: {np.mean(np.abs(clustering_samples-clustering_true)/clustering_true)} +/- {np.std(np.abs(clustering_samples-clustering_true)/clustering_true)}\n")
f.write(f"Average relative RMSE for clustering_dr:      {np.mean(np.abs(clustering_dr-clustering_true)/clustering_true)} +/- {np.std(np.abs(clustering_dr-clustering_true)/clustering_true)}\n")
f.write(f"Average relative RMSE for degree_dr:      {np.mean(np.abs(degree_dr-degree_true)/degree_true)} +/- {np.std(np.abs(degree_dr-degree_true)/degree_true)}\n")
f.write(f"Average relative RMSE for degree_TM_VM:   {np.mean(np.abs(degree_TM_VM-degree_true)/degree_true)} +/- {np.std(np.abs(degree_TM_VM-degree_true)/degree_true)}\n")
f.close()
