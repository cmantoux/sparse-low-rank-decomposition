"""
This file reproduces the experiment on a simulated data set of simple networks in section 4.1.

The networks have 100 nodes split in 5 communities. An additional block of interactions between
individuals [k,...,l] and [p,...,q] is created for each sample network. Finally, 20% of the edges
are randomly flipped from 0 to 1 and conversely.


REMARK
When running the code on different platforms, we noticed that the code produced
slightly different results, due to differences in the distributions of numpy.
In order to make the code yield identical results on different platforms,
we round off the parameters optimization objective to 6 decimal places.
"""

import numpy as np
from matplotlib.pyplot import *
from tqdm import *
import skopt

from src.utils import *
import src.douglas_rachford as dr

np.random.seed(0)


# ==================================================
# ========= GLOBAL EXPERIMENT PARAMETERS ===========
# ==================================================


n = 100        # number of nodes
clusters = 5   # number of clusters
n_samples = 10 # samples count

theta_dr = 0.1 # Douglas-Rachford global step size
tau_dr = 0.1   # Douglas-Rachford proximal step size
niter = 400    # Number of optimization iterations


# ==================================================
# ================ DATA GENERATION =================
# ==================================================

print("Generating the data set.")

# Define the template matrix, i.e. the clusters
sep = np.cumsum(np.random.rand(clusters))
sep /= sep[-1]
sep = (n*sep).astype(np.int)
sep = [0] + list(sep)
T_init = np.zeros((n,n))
for i in range(clusters):
    T_init[sep[i]:sep[i+1], sep[i]:sep[i+1]] = 1


# Generate block perturbations
Ds_init = np.zeros((n_samples,n,n))
for i in range(n_samples):
    while True:
        k, l = sorted(np.random.randint(n, size=2))
        p, q = sorted(np.random.randint(n, size=2))
        if k!=l and p!=q and T_init[k:l, p:q].mean()!=1:
            break
    Ds_init[i, k:l, p:q] = 1

samples = np.maximum(np.tile(T_init[None], (n_samples,1,1)), Ds_init)
Vs_init = samples - np.tile(T_init[None], (n_samples,1,1))
V_init = np.vstack(Vs_init)

perturbation = (np.random.rand(10,100,100) < 0.2)
samples[perturbation] = 1-samples[perturbation]

M = np.mean(samples, axis=0)


# ========================================================
# ================ PARAMETER SELECTION ===================
# ========================================================

# This step is skipped here, for the full process run simple_example.py

parameters = [48.47, 0.46, 1.14, 0.35]

print("Using pre-selected parameters:")
print("lambda = ", parameters[0])
print("rho    = ", parameters[1])
print("mu     = ", parameters[2])
print("nu     = ", parameters[3])


# ========================================================
# ====================== RESULTS =========================
# ========================================================

# Train model with selected parameters
model = dr.SparseLowRank(*parameters, theta=theta_dr, tau=tau_dr, niter=niter, progress=False)
model.fit(samples)

# Create and save matplotlib figure
figure(figsize=(18,8), dpi=100)
subplots_adjust(wspace=0.01, left=0.01, right=0.99, bottom=0.05, top=0.95)

subplot(2,4,1)
title("Sample $A_1$", fontsize=22)
axis("off")
imshow(samples[0], vmin=-0.1, vmax=1.1)
colorbar()

subplot(2,4,2)
title("Template $T$", fontsize=22)
axis("off")
imshow(T_init, vmin=-0.1, vmax=1.1)
colorbar()

subplot(2,4,3)
title("Deviation $V_1$", fontsize=22)
axis("off")
imshow(Vs_init[0], vmin=-0.1, vmax=1.1)
colorbar()

subplot(2,4,4)
title("Uncorrupted $T+V_1$", fontsize=22)
axis("off")
imshow(T_init + Vs_init[0], vmin=-0.1, vmax=1.1)
colorbar()

subplot(2,4,5)
title("Mean sample", fontsize=22)
axis("off")
imshow(M, vmin=-0.1, vmax=1.1)
colorbar()

subplot(2,4,6)
title("Estimated $T$", fontsize=22)
axis("off")
imshow(model.S, vmin=-0.1, vmax=1.1)
colorbar()

subplot(2,4,7)
title("Estimated $V_1$", fontsize=22)
axis("off")
imshow(model.Ws[0], vmin=-0.1, vmax=1.1)
colorbar()

subplot(2,4,8)
title("Estimated $T+V_1$", fontsize=22)
axis("off")
imshow(model.S+model.Ws[0], vmin=-0.1, vmax=1.1)
colorbar()

savefig("figures/simple_example.pdf")

print("Saved figure in figures/simple_example.pdf.")

# Compute the proportion of the support that was recovered correctly
# up to a certain threshold

def hard_threshold(A, t):
    """
    Removes all coefficients from A with absolute value under t
    """
    B = A.copy()
    B[np.abs(B) <= t] = 0
    return B

print("Support recovery:")

s = (hard_threshold(model.T, 0.1) == 0)
t = (T_init == 0)
print("Proportion of coefficients in T with correctly recovered support (within threshold 0.1):", (s==t).mean())

s = (hard_threshold(model.V, 0.1) == 0) # which coefficient is null in the estimated V up to threshold 0.1
t = (V_init == 0) # which coefficient is null in the true V up to threshold 0.1
print("Proportion of coefficients in V with correctly recovered support (within threshold 0.1):", (s==t).mean())
