"""
This file contains functions to generate sparse low rank matrices and data sets as used in the paper.
The main functions are sparse_low_rank and dataset.
"""


import numpy as np


def sparse_low_rank_(n, d, sparsity, positive=False, symmetric=False):
    """
    Auxiliary function to generate a square sparse low rank matrix X = UDV by drawing U, D, and V.
    
    Input:
        - n: matrix size
        - d: matrix rank
        - sparsity:  percentage of null coefficients in U and V
        - positive:  if True, U and V have positive coefficients
        - symmetric: if True, U=V
    Output:
        - X with shape (n,n), rank d and expected proportion of null coefficients equal to sparsity
    """
    
    T = np.zeros((n,n))
    while np.linalg.matrix_rank(T) != d:
        # While U and V have null rows
        while True:
            if positive:
                U = np.abs(np.random.randn(n, d))
            else:
                U = np.random.randn(n, d)
            # Induce sparsity
            rU = np.random.rand(n, d)
            U = U * (rU > sparsity)
            if np.linalg.matrix_rank(U)>=d and np.linalg.norm(U, axis=0).min() > 0:
                break
        
        # If not symmetric, generate V too
        if not symmetric:
            while True:
                if positive:
                    V = np.abs(np.random.randn(n, d))
                else:
                    V = np.random.randn(n, d)
                # Induce sparsity
                rV = np.random.rand(n, d)
                V = V * (rV > sparsity)
                if np.linalg.matrix_rank(V)>=d and np.linalg.norm(V, axis=0).min() > 0:
                    break

        if positive:
            D = np.diag(np.abs(np.random.randn(d)))
        else:
            D = np.diag(np.random.randn(d))
        
        if symmetric:
            T = U@D@U.T
        else:
            T = U@D@V.T
    return T


def sparse_low_rank(n, d, sparsity, positive=False, symmetric=False):
    """
    Generates a square sparse low rank matrix X = UDV by drawing U, D, and V, with
    desired rank and sparsity.
    
    Input:
        - n: matrix size
        - d: matrix rank
        - sparsity:  percentage of null coefficients in X
        - positive:  if True, U and V have positive coefficients
        - symmetric: if True, U=V
    Output:
        - X with shape (n,n), rank d and expected proportion of null coefficients equal to sparsity
    
    Correct the proportion of sparse coefficients obtained in sparse_low_rank_ 
    """
    
    threshold = 1-np.sqrt(1-sparsity**(1/d)) # See Lemma 1 in the paper's supplementary material
    return sparse_low_rank_(n, d, threshold, positive, symmetric)


def dataset(n, n_samples, T_rank, T_sparse, T_positive, V_rank, V_sparse, V_positive,
            noise_amplitude=1, noise_sparsity=0.8, noise_positive=False,
            symmetric=False, T_init=None):
    """
    Generates a set of n_samples adjacency matrices with size (n,n), under the model:
    samples[i] = T_init + Vs_init[i] + eps[i]
    T_init and Vs_init[i] are sparse low rank matrices, and eps is sparse.
    
    Input:
        - n: matrix size
        - n_sample: number of model samples
        - T_rank: rank of the template T
        - V_rank: rank of the deviations V
        - T_positive: if True, T has positive coefficients
        - V_positive: if True, V has positive coefficients
        - T_sparse: expected proportion of null coefficients in T
        - V_sparse: expected proportion of null coefficients in V
        - noise_amplitude: noise standard deviation
        - noise_sparsity:  expected proportion of null coefficients in the noise
        - noise_positive:  if True, the noise will be computed as max(Sparse Gaussian, -T-V[i])
                                so that the sample is positive. The noise may thus have
                                negative coefficients. Otherwise the noise is just a sparse Gaussian noise.
        - symmetric: if True, then T, V, and the noise produced are symmetric
        - T_init: if not None, the data set is generated using the provided template
                    instead of generating one.
    Output:
        - samples with shape (n_samples,n,n): list of model samples
        - T_init with shape (n,n): sparse low rank template
        - Vs_init with shape (n_samples,n,n): sparse low rank deviations
        - eps with shape (n_samples,n,n): sparse noise
    """
    
    if T_init is None:
        T_init = sparse_low_rank(n, T_rank, T_sparse, positive=T_positive, symmetric=symmetric)
    if symmetric:
        idx = np.tril_indices(n)
    samples = np.zeros((n_samples, n, n))
    Vs_init = []
    eps = []
    for i in range(n_samples):
        Vi = sparse_low_rank(n, V_rank, V_sparse, positive=V_positive, symmetric=symmetric)
        Vs_init.append(Vi)
        
        noise = np.random.randn(n,n)
        if noise_positive:
            noise = np.maximum(noise, -T_init-Vs_init[i])
        rN = np.random.rand(n,n)
        noise = noise * (rN > noise_sparsity)
        noise = noise_amplitude * noise
        if symmetric:
            noise[idx[0], idx[1]] = noise[idx[1], idx[0]]
        samples[i] = T_init + Vs_init[i] + noise
        eps.append(noise)
    return samples, T_init, Vs_init, eps