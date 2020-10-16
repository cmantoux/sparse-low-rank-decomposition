"""
This file reproduces the results on US domestic flight networks of section 5.1.
"""

print("Importing modules.")

import numpy as np
import pandas as pd
from matplotlib.pyplot import *
import networkx as nx
import cpalgorithm as cp # module for core-periphery structure detection

from src.utils import *
import src.douglas_rachford as dr


# =======================================================
# =========== GLOBAL EXPERIMENT PARAMETERS ==============
# =======================================================

# Parameters for the sparse low rank decomposition with L1 loss
lambd = 10  # Low rank penalty for the template
rho   = 0.1 # Sparsity penalty for the template
mu    = 5   # Low rank penalty for the deviations
nu    = 0.1 # Sparsity penalty for the deviations

# Parameters for the Douglas-Rachford optimization
theta = 0.9 # Global step size
tau   = 0.1 # Proximal step size
niter = 200 # Number of iterations


# =====================================================
# ================ DATA PREPROCESSING =================
# =====================================================

print("Preprocessing the data set.")

# list of all flights for 10 days
df = pd.read_csv("data/flightsFAA_stnet.csv")
df_numpy = df.values[:,:3]
df_numpy[:,2] = [int(x[1:]) for x in df_numpy[:,2]]
df_numpy = df_numpy.astype(int)

T = 480 # total number of hours
n = 299 # total number of cities/graph nodes

adj = np.zeros((T,n,n)) # Adjacency matrix for each hour
idx = np.where(np.bincount(df_numpy[:,[0,1]].reshape(-1))!=0)[0]
dico = {i: k for k, i in enumerate(idx)}

for i in range(df_numpy.shape[0]):
    a, b, t = df_numpy[i]
    adj[t-1, dico[a], dico[b]] += 1

# coordinates for each airport
df2 = pd.read_csv("data/flightsFAA_coords.csv")


# Adjacency matrices for all 10 days
# samples has shape (10,299,299)
samples = np.array(np.array_split(adj, 10)).sum(axis=(1))


def rearrange(samples, order):
    """
    Permute the nodes in an adjacency matrix or an array of adjacency matrices.
    """
    return np.array(samples)[...,order,:][...,:,order]


# ========================================================
# ================ PERFORM DECOMPOSITION =================
# ========================================================

print("Performing sparse low rank decomposition with L1 loss.")

model = dr.SparseLowRank(lambd=lambd, rho=rho, mu=mu, nu=nu, theta=theta, tau=tau, niter=niter, progress=True)
model.fit(samples)


# ========================================================
# ====================== RESULTS =========================
# ========================================================


print("Relative error between the estimate T+V_i and the samples A_i:")
print(relative_error(np.tile(model.S[None,:,:], (10,1,1))+model.Ws, samples))
# Here use the sparse versions of the result, model.S and model.W (instead of model.T and model.V)
# denoted T_1 and V_1 in the paper, as the true adjacency matrix are very sparse.

rank = np.linalg.matrix_rank
print("Template rank:", rank(model.T))
print("Deviations maximum rank:", rank(model.Vs).max())
print("Samples maximum rank:", rank(samples).max())
print("Mean sample rank:", rank(samples.mean(axis=0)))

sparsity = lambda A: (np.array(A)==0).mean() # proportion of null coefficients
print("Template sparsity:", sparsity(model.S))
print("Deviations sparsity:", sparsity(model.Ws))
print("Samples sparsity:", sparsity(samples))
print("Mean sample sparsity:", sparsity(samples.mean(axis=0)))

# Detect the core/periphery structure using the module cpalgorithm
algo = cp.LowRankCore(beta=0.1)
algo.detect(nx.DiGraph(model.S))
mod = np.array(list(algo.get_coreness().values()))
core = np.where(mod==1)[0]      # list of core nodes
periphery = np.where(mod==0)[0] # list of periphery nodes
periphery_core = np.concatenate([periphery, core])



figure(figsize=(14,4), dpi=100)
subplots_adjust(wspace=0.0, left=0.0, right=0.99, bottom=0.05, top=0.9)

subplot(1,3,1)
title("Template", fontsize=15)
imshow(rearrange(model.T, periphery_core))
axis("off")
colorbar()

subplot(1,3,2)
title("Deviation $V_1$", fontsize=15)
imshow(rearrange(model.Vs[0], periphery_core))
colorbar()
axis("off")

subplot(2,6,5)
title("Template (core)", fontsize=15)
imshow(rearrange(model.T, core))
axis("off")
colorbar()

subplot(2,6,6)
title("Deviation $V_1$ (core)", fontsize=15)
imshow(rearrange(model.Vs[0], core))
colorbar()
axis("off")

subplot(2,6,11)
title("Deviation $V_2$ (core)", fontsize=15)
imshow(rearrange(model.Vs[1], core))
colorbar()
axis("off")

subplot(2,6,12)
title("Deviation $V_3$ (core)", fontsize=15)
imshow(rearrange(model.Vs[2], core))
colorbar()
axis("off")

savefig("figures/airline_networks.pdf")

print("Saved figure in figure/airline_networks.pdf.")


def locate(i):
    """Geographical coordinates (longitude, latitude) of core node i"""
    return df2.loc[df2["node"]==idx[core[i]]]


print("\n\nCoordinates (longitude, latitude) of nodes mentioned in the paper:\n")
print("For the two bright spots in deviation V_1:")
print("7th core node (Denver):", locate(6))
print("15th core node (New York's eastern airport):", locate(14))
print("\nFor the dark pattern in V_1:")
print("10th core node (New York's western airport):", locate(9))
