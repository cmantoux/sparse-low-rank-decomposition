"""
This script reproduces the results of section 4.2 on simulated data with unconstrained signs.
It is divided into separate sections.
- Data generation
- SPLRD model (ours): parameter selection
- SPLRD model (ours): evaluation
- SPLR estimation (Richard et al. 2012) on T: parameter selection
- SPLR estimation (Richard et al. 2012) on T: evaluation
- SPLR estimation (Richard et al. 2012) on V: parameter selection
- SPLR estimation (Richard et al. 2012) on V: evaluation
The procedures for each model are very similar.

Generally, the naming conventions in the code go as follows:
- [...]_init is the generated uncorrupted data:
    - T_init is the true template
    - Vs_init is the true deviations
    - V_init is the true deviations, stacked in a tall matrix.
- model.T and model.V refer to the estimates computed through the optimization.
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

niter = 200 # Number of iterations for both algorithms


# ========================================================
# ================ DATA GENERATION     ===================
# ================ Unconstrained signs ===================
# ========================================================

print("\nGenerating the data sets.")
np.random.seed(0)

# Generate train data with unconstrained signs

samples_train = np.zeros((n_train, n_samples, n, n)) # Model samples
T_init_train = np.zeros((n_train, n, n))             # Template matrices
M_train = np.zeros((n_train, n, n))                  # Mean samples
Vs_init_train = np.zeros((n_train, n_samples, n, n)) # Deformations matrices
V_init_train = np.zeros((n_train, n_samples * n, n)) # Stacked deformations

for k in range(n_train):
    samples_train[k], T_init_train[k], Vs_init_train[k], _ = dataset(n, n_samples,
                                                            T_rank=10, T_sparse=0.7, T_positive=False,
                                                            V_rank=10, V_sparse=0.7, V_positive=False,
                                                            noise_sparsity=0.7, noise_amplitude=1, noise_positive=False,
                                                           )
    M_train[k] = samples_train[k].mean(axis=0)
    V_init_train[k] = np.vstack(Vs_init_train[k])

# Generate test data with unconstrained signs

samples_test = np.zeros((n_test, n_samples, n, n)) # Model samples
T_init_test = np.zeros((n_test, n, n))             # Template matrices
M_test = np.zeros((n_test, n, n))                  # Mean samples
Vs_init_test = np.zeros((n_test, n_samples, n, n)) # Deformations matrices
V_init_test = np.zeros((n_test, n_samples * n, n)) # Stacked deformations

for k in range(n_test):
    samples_test[k], T_init_test[k], Vs_init_test[k], _ = dataset(n, n_samples,
                                                            T_rank=10, T_sparse=0.7, T_positive=False,
                                                            V_rank=10, V_sparse=0.7, V_positive=False,
                                                            noise_sparsity=0.7, noise_amplitude=1, noise_positive=False,
                                                           )
    M_test[k] = samples_test[k].mean(axis=0)
    V_init_test[k] = np.vstack(Vs_init_test[k])


# Estimation error between M and T
errors_M = np.array([relative_error(M_test[k], T_init_test[k]) for k in range(n_test)])
# Estimation error between A_i-M and V_i
errors_A_M = np.array([relative_error(samples_test[k]-M_test[k][None,:,:],
                                      Vs_init_test[k]) for k in range(n_test)])

f = open(f"{log_dir}/mean_unconstrained.txt", "a")
f.writelines([
    f"Average relative RMSE between the mean sample and the template: {np.mean(errors_M)} +/- {np.std(errors_M)}\n",
    f"Average relative RMSE between A_i-M and V_i: {np.mean(errors_A_M)} +/- {np.std(errors_A_M)}\n"
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
        m.fit(samples_train[k])            # Perform the decomposition on each train data set
        # We compute the relative error between the estimated V_i and the true V_i:
        result += relative_error(m.V, V_init_train[k])

    # Write step result
    f = open(f"{log_dir}/L1_unconstrained.txt", "a")
    f.write(f"===== Training step {cnt.k} =====\n")
    f.write(f"Parameters: "+str(params)+"\n")
    f.write(f"Total training relative RMSE: {result}\n")
    f.close()

    # Update progress bar
    progress.update(1)
    progress.set_postfix({"objective": result})
    return np.round(result, 6)

# Execute the parameter optimization
results_dr = skopt.forest_minimize(objective, SPACE, n_calls=100, random_state=0)

# Save the results
f = open(f"{log_dir}/L1_unconstrained.txt", "a")
f.write(f"\n\nBest training relative RMSE: {np.round(results_dr['fun'], 4)}\n")
f.write(f"Best training parameters: "+str(results_dr['x'])+"\n")
f.close()


# ========================================================
# ============== SPLRD MODEL WITH L1 LOSS ================
# ============== Evaluation on test data  ================
# ========================================================

print("\nL1 model: evaluation")

models_dr_test = []

# Estimate the template and deviations on each test data set
for k in trange(n_test):
    models_dr_test.append(dr.SparseLowRank(*results_dr['x'], theta=theta_dr, tau=tau_dr, progress=False, niter=niter))
    models_dr_test[k].fit(samples_test[k])

# Compute the reconstruction errors
# Between the estimated template and the true template
errors_dr_T   = [relative_error(models_dr_test[k].T, T_init_test[k]) for k in range(n_test)]
# Between the estimated deviation and the true deviation
errors_dr_V   = [relative_error(models_dr_test[k].V, V_init_test[k]) for k in range(n_test)]
# Between (A_i-[estimated template]) and the true deviation
errors_dr_A_T = [relative_error(samples_test[k]-models_dr_test[k].T[None,:,:], Vs_init_test[k]) for k in range(n_test)]


# Save the results
f = open(f"{log_dir}/L1_unconstrained_test.txt", "a")
f.writelines([
    "Best parameters for the L1 model with unconstrained sign:\n",
    f"lambda = {np.round(results_dr['x'][0], 3)}\n",
    f"rho    = {np.round(results_dr['x'][1], 3)}\n"
    f"mu     = {np.round(results_dr['x'][2], 3)}\n"
    f"nu     = {np.round(results_dr['x'][3], 3)}\n \n",
    f"Average relative RMSE for the template  T = {np.mean(errors_dr_T)} +/- {np.std(errors_dr_T)}\n",
    f"Average relative RMSE for the deviations V = {np.mean(errors_dr_V)} +/- {np.std(errors_dr_V)}\n",
    f"Average relative RMSE for the deviations V estimated by A_i-T = {np.mean(errors_dr_A_T)} +/- {np.std(errors_dr_A_T)}\n",
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

# Define optimization space
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
        # Perform sparse low rank denoising from Richard et al. (2012) on the mean sample M :
        TM_train = dr.single_sample_splr(M_train[k], params['lambd'], params['rho'],
                                         theta_dr, tau_dr, niter)
        result += relative_error(TM_train, T_init_train[k])

    # Save step result
    f = open(f"{log_dir}/splr_T_unconstrained.txt", "a")
    f.write(f"===== Training step {cnt.k} =====\n")
    f.write(f"Parameters: "+str(params)+"\n")
    f.write(f"Total training relative RMSE: {result}\n")
    f.close()

    # Update progress bar
    progress.update(1)
    progress.set_postfix({"obj": result})
    return np.round(result, 6)

# Execute parameter optimization
results_denoise_T = skopt.forest_minimize(objective_denoise, SPACE_DENOISE, n_calls=100, random_state=0)

f = open(f"{log_dir}/splr_T_unconstrained.txt", "a")
f.write(f"\n\nBest training relative RMSE: {np.round(results_denoise_T['fun'], 4)}\n")
f.write(f"Best training parameters: "+str(results_denoise_T['x'])+"\n")
f.close()


# ===========================================================
# ================ SPLR DENOISING FOR T    ==================
# ================ Evaluation on test data ==================
# ===========================================================

print("\nSPLR T^M: evaluation")

TM_test = np.zeros((n_test, n, n))

for k in trange(n_test):
    TM_test[k] = dr.single_sample_splr(M_test[k], *results_denoise_T['x'], theta_dr, tau_dr, niter)

# Compute the reconstruction error for T^M:
errors_dr_TM = [relative_error(TM_test[k], T_init_test[k]) for k in range(n_test)]

# Save the results
f = open(f"{log_dir}/splr_T_unconstrained_test.txt", "a")
f.writelines([
    "Best parameters for T^M:\n",
    f"lambda = {np.round(results_denoise_T['x'][0], 3)}\n",
    f"rho    = {np.round(results_denoise_T['x'][1], 3)}\n\n"
    f"Average relative RMSE for the template  T = {np.mean(errors_dr_TM)} +/- {np.std(errors_dr_TM)}\n",
             ])
f.close()


# ========================================================
# ================ SPLR DENOISING FOR V ==================
# ================ Parameter selection  ==================
# ========================================================

# In this section, we compute what we denoted V^M in the paper
# We choose the parameters that best reconstruct the deviations V

print("\nSPLR V^M: parameters selection")


np.random.seed(0)

# Define optimization space
SPACE_DENOISE = [
    skopt.space.Real(0, 10, name='mu', prior='uniform'),
    skopt.space.Real(0, 10, name='nu', prior='uniform'),
]

# First, we compute T^M with the best parameters obtained in the previous code section
TM_train = np.zeros_like(T_init_train)
for k in range(n_train):
    TM_train[k] = dr.single_sample_splr(M_train[k], *results_denoise_T['x'], theta_dr, tau_dr, niter)

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
            V[i*n:(i+1)*n] = dr.single_sample_splr(samples_train[k,i]-TM_train[k],
                                                   params['mu'], params['nu'], theta_dr, tau_dr, niter)
        result += relative_error(V, V_init_train[k])

    f = open(f"{log_dir}/splr_V_unconstrained.txt", "a")
    f.write(f"===== Training step {cnt.k} =====\n")
    f.write(f"Parameters: "+str(params)+"\n")
    f.write(f"Total training relative RMSE: {result}\n")
    f.close()

    progress.update(1)
    progress.set_postfix({"obj": result})
    return np.round(result, 6)

# Execute parameter optimization
results_denoise_V = skopt.forest_minimize(objective_denoise, SPACE_DENOISE, n_calls=100, random_state=0)

f = open(f"{log_dir}/splr_V_unconstrained.txt", "a")
f.write(f"\n\nBest training relative RMSE: {np.round(results_denoise_V['fun'], 4)}\n")
f.write(f"Best training parameters: "+str(results_denoise_V['x'])+"\n")
f.close()


# ===========================================================
# ================ SPLR DENOISING FOR V    ==================
# ================ Evaluation on test data ==================
# ===========================================================

print("\nSPLR V^M: evaluation")


VM_test = np.zeros((n_test, n_samples, n, n))

# Perform the decomposition for each V_i in each test data set
for k in trange(n_test):
    for i in range(n_samples):
        VM_test[k,i] = dr.single_sample_splr(samples_test[k,i]-TM_test[k], *results_denoise_V['x'], theta_dr, tau_dr, niter)

# Compute the reconstruction error for V^M
errors_dr_VM = [relative_error(VM_test[k], Vs_init_test[k]) for k in range(n_test)]

# Save the results
f = open(f"{log_dir}/splr_V_unconstrained_test.txt", "a")
f.writelines([
    "Best parameters for V^M:\n",
    f"mu    = {np.round(results_denoise_V['x'][0], 3)}\n",
    f"nu    = {np.round(results_denoise_V['x'][1], 3)}\n\n"
    f"Average relative RMSE for the deviations V = {np.mean(errors_dr_VM)} +/- {np.std(errors_dr_VM)}\n",
             ])
f.close()
