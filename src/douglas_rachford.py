"""
This file contains the implementations of the Douglas-Rachford algorithm for
- The sparse low rank decomposition A_i = T+V_i
- The sparse low rank denoising with L1 loss for one single matrix as proposed by Richard et al. (2012).

The duplicate variables for the SPLR decomposition are named as follows:
- T_* in the paper is denoted T in the code,
- T_1 in the paper is denoted S in the code,
- V_* in the paper is denoted V in the code,
- V_1 in the paper is denoted W in the code,
- W   in the paper is denoted Y in the code.

The naming convention for the duplicate variables of the SPLR denoising on a single matrix (Richard et al.) is as follows:
- A is the variable for the low rank penalty,
- B is the variable for the sparse penalty,
- C is the variable for the data attachment term.

In both cases, all the duplicate variables are stacked in a single matrix X on which the Douglas-Rachford is applied.

The samples A_i are stored in an array samples with size (n_samples, n, n).

Please note that the notation conventions in the code are slightly different from those in the paper:
In the code n stands for the number of nodes and n_samples for the number of networks in the data set.
"""

import numpy as np
from tqdm import *
import src.proximal as prox


# ===========================================================================
# ================ GENERAL PURPOSE DOUGLAS-RACHFORD ALGORITHM ===============
# ===========================================================================


def rprox(prox):
    """
    Given a proximal operator, compute the related reversed proximal operator.

    Input:
        - prox: function taking arguments (X, *args), with X the main variable
                and args the function parameters
    Output:
        - function f such that f(x, *args) = 2*x - prox(x, *args)
    """

    def f(*args):
        return 2*prox(*args)-args[0]
    return f


def DR(prox1, prox2, theta, args, x0, niter, progress=False, objective=None):
    """
    General-purpose implementation of the Douglas-Rachford algorithm to mimimize
    a function f(x)+g(x).

    Input:
        - prox1, prox2: proximal operators for f and g. The syntax to call them is prox(variable, *arguments)
        - theta: step size
        - args: arguments for both proximal operators
        - x0: numpy array for the initial value
        - niter: maximum iteration
        - progress: if set to True, the iterations will show on a tqdm progress bar.
        - objective: the objective function to be minimized. Used only for monitoring when progress is True.

    Output:
        - The output of the Douglas-Rachford algorithm, with same shape than x0.
    """

    y = x0
    # Define the reversed proximal operators:
    rp1 = rprox(prox1)
    rp2 = rprox(prox2)
    obj = np.inf
    if progress:
        it = trange(niter)
    else:
        it = range(niter)
    for i in it:
        y_old = y
        tmp = rp1(y, *args)
        y = (1-theta)*y + theta*rp2(tmp, *args)
        diff = np.linalg.norm(y-y_old, ord=np.inf)
        if progress and i%10==0 and objective is not None:
            obj_old = obj
            obj = objective(prox1(y, *args), *args)
            it.set_postfix({"objective":  obj})
        # Stopping criterion (rarely met in practice):
        if diff < 0.1:
            break
    return prox1(y, *args)


# =================================================================
# ================ L1 LOSS MODEL                     ==============
# ================ Proximal operators  and objective ==============
# =================================================================


def getTSVYW(X):
    """
    Split the global variable X.

    Input:
        - X with shape ((2+3*n_samples)*n, n)
    Output:
        - T with shape (n,n)
        - S with shape (n,n)
        - V with shape (n_samples*n,n)
        - Y with shape (n_samples*n,n)
        - W with shape (n_samples*n,n)
    """

    n = X.shape[1]
    n_samples = (X.shape[0]-2*n)//(3*n)
    T = X[:n]
    S = X[n:2*n]
    V = X[2*n:n*n_samples+2*n]
    Y = X[n*n_samples+2*n:2*n*n_samples+2*n]
    W = X[2*n*n_samples+2*n:]
    return T, S, V, Y, W


def proj_constraints(X, *args):
    """
    Project (T, S, V, Y, W) onto the subspace {T=S, V=W, T+V=Y}.

    Input:
        - X with shape ((2+3*n_samples)*n, n)
        - args is not used here
    Output:
        - X with shape ((2+3*n_samples)*n, n) respecting the equality constraints
    """

    n = X.shape[1]
    n_samples = (X.shape[0]-n)//(3*n)
    T, S, V, Y, W = getTSVYW(X)
    C = 1/(2*n_samples+6)

    Vbar = np.sum(np.array_split(V, n_samples), axis=0)*2*C
    Ybar = np.sum(np.array_split(Y, n_samples), axis=0)*2*C
    Wbar = np.sum(np.array_split(W, n_samples), axis=0)*2*C

    T2 = 3*C*(T+S) + Ybar - Vbar/2 - Wbar/2
    S2 = T2.copy()
    x = -C*(T+S) - Ybar/3 + Wbar/6 + Vbar/6
    V2 = (V+W+Y)/3 + np.tile(x, (n_samples,1))
    Y2 = V2 + np.tile(T2, (n_samples,1))
    W2 = V2.copy()
    return np.concatenate((T2, S2, V2, Y2, W2))


def prox_separate(X, samples, lambd, rho, mu, nu, tau):
    """
    Proximal operators for T, S, V, Y, W applied entry-wise.

    Input:
        - X with shape ((2+3*n_samples)*n, n): global optimization variable
        - lambd: Rank regularization parameter for T
        - rho:   Sparsity regularization parameter for T (S in the code)
        - mu:    Rank regularization parameter for V
        - nu:    Sparsity regularization parameter for (W in the code)
        - tau:   proximal step size
    Output:
        - X with shape ((2+3*n_samples)*n, n) after proximal step
    """

    n = X.shape[1]
    n_samples = (X.shape[0]-n)//(3*n)
    T, S, V, Y, W = getTSVYW(X)

    T = prox.prox_norm_nuclear(T, t=lambd*tau) # Nuclear norm proximal operator
    S = prox.prox_norm_1(S, t=rho*tau)      # L1 norm proximal operator
    Y = prox.prox_norm_1(Y, t=tau, offset=np.vstack(samples)) # L1 proximal operator with offset
    for i in range(n_samples):
        V[i*n:(i+1)*n] = prox.prox_norm_nuclear(V[i*n:(i+1)*n], t=mu*tau)
    W = prox.prox_norm_1(W, t=nu*tau)
    return np.concatenate((T, S, V, Y, W))


def objective(X, samples, lambd, rho, mu, nu, *args):
    """
    Objective function for SPLR the decomposition with L2 norm.

    Input:
        - X with shape ((2+3*n_samples)*n, n): global optimization variable
        - samples with shape (n_samples, n, n)
        - lambd: Rank regularization parameter for T
        - rho:   Sparsity regularization parameter for T (S in the code)
        - mu:    Rank regularization parameter for V
        - nu:    Sparsity regularization parameter for (W in the code)
    Output:
        - Objective function defined in equation (SPLRD) with the L1 loss
    """

    T, S, V, Y, W = getTSVYW(X)
    n = T.shape[0]
    n_samples = (X.shape[0]-n)//(3*n)
    obj = 0
    obj += lambd * np.linalg.norm(T, ord='nuc')
    obj += rho * np.linalg.norm(S, ord=1)
    obj += np.linalg.norm(Y-np.vstack(samples), ord=1)
    for i in range(n_samples):
        obj += mu * np.linalg.norm(V[i*n:(i+1)*n], ord='nuc')
    obj += nu * np.linalg.norm(W, ord=1)
    return obj


class SparseLowRank(object):
    """
    Python class for the sparse low rank decomposition with L1 loss.
    Used to perform and store the optimization results in a single object.
    """

    def __init__(self, lambd, rho, mu, nu, theta, tau, niter, progress=False):
        """
        Initialize the decomposition model.

        Input:
            - lambd: Rank regularization parameter for T
            - rho:   Sparsity regularization parameter for T (S in the code)
            - mu:    Rank regularization parameter for V
            - nu:    Sparsity regularization parameter for (W in the code)
            - theta: Douglas-Rachford step size
            - tau:   proximal step size
            - niter: maximum iterations in Douglas-Rachford
            - progress: if True, Douglas-Rachford with display as a tqdm progress bar.
        """
        self.theta = theta
        self.lambd = lambd
        self.rho = rho
        self.mu = mu
        self.nu = nu
        self.tau = tau
        self.niter = niter
        self.progress = progress

        """
        V and Vs contain the same information.
        V has shape (n*n_samples, n) whereas Vs has shape (n_samples, n, n).
        The same goes for W and Y.
        """
        self.T = None
        self.S = None
        self.V = None
        self.Vs = None
        self.W = None
        self.Ws = None
        self.Y = None
        self.Ys = None

    def fit(self, samples, X_init=None):
        """
        Perform the optimization to compute T, S, V, Y, W.

        Input:
            - samples with shape (n_samples, n, n)
            - X_init (optional) with shape ((2+3*n_samples)*n, n) initial value for the optimizer
        Output:
            - All the outputs (T, S, V, Vs, Y, Ys, W, Ws) are stored in the object.
        """

        n = samples.shape[1]
        n_samples = samples.shape[0]

        if X_init is None:
            T = samples.mean(axis=0)
            V = np.vstack(samples)-np.tile(T, (n_samples, 1))
            X_init = np.concatenate((T,T,V,V,V))

        # Using prox_separate as first proximal operator allows
        # to get the unprojected version of the optimization result.
        XDR = DR(prox_separate, proj_constraints,
                 self.theta,
                 (samples, self.lambd, self.rho,
                  self.mu, self.nu, self.tau),
                 X_init, self.niter,
                 objective=objective,
                 progress=self.progress
                )

        self.T, self.S, self.V, self.Y, self.W = getTSVYW(XDR)
        self.Vs = np.array_split(self.V, n_samples)
        self.Ws = np.array_split(self.W, n_samples)
        self.Ys = np.array_split(self.Y, n_samples)


# ==========================================================================
# ========= SINGLE SAMPLE SPLR WITH L1 LOSS (Richard et al., 2012) =========
# ========= Proximal operators and objective                       =========
# ==========================================================================


def prox_denoise(X, base, lambd, rho, tau):
    """
    Disjoint proximal operators for A, B, C, applied entry-wise.

    Input:
        - X with shape (3*n,n): optimization variable
        - base with shape (n,n): input matrix to be denoised
        - lambd: rank regularization parameter
        - rho:   sparsity regularization parameter
        - tau:   proximal step size
    Output:
        - X with shape (3*n,n) after proximal step size
    """

    A, B, C = np.array_split(X, 3)
    A = prox.prox_norm_nuclear(A, t=tau*lambd)
    B = prox.prox_norm_1(B, t=tau*rho)
    C = prox.prox_norm_1(C, t=tau, offset=base)
    return np.concatenate([A,B,C])


def prox_mean(X, *args):
    """
    Project (A, B, C) onto the set {A=B=C}.

    Input:
        - X with shape (3*n,n): optimization variable
        - *args is not used here.
    """

    A, B, C = np.array_split(X, 3)
    D = (A+B+C)/3
    return np.concatenate([D]*3)


def objective_single_sample(X, base, lambd, rho, tau):
    """
    Optimization objective for single sample SPLR estimation.

    Input:
        - X with shape (3*n,n): optimization variable
        - base with shape (n,n): input matrix to be denoised
        - lambd: rank regularization parameter
        - rho:   sparsity regularization parameter
        - tau:   proximal step size (not used here)
    Output:
        - Optimization defined in equation (SPLR) in the paper.
    """

    A, B, C = np.array_split(X, 3)
    obj = lambd*np.linalg.norm(A, 'nuc')
    obj += rho*np.linalg.norm(B, 1)
    obj += tau*np.linalg.norm(C-base, 1)
    return obj

def single_sample_splr(X, lambd, rho, theta, tau, niter):
    """
    Perform the sparse low rank denoising with L1 loss from Richard et al. (2012) using Douglas-Rachford.
    Here we return directly the mean of the duplicate variables.
    In order to quantitatively assess the rank and sparsity of the solution, consider
    returning (A, B, C) instead of the mean. A has low rank and B is sparse (both are extremely close).

    Input:
        - X with shape (n,n): input matrix to be denoised
        - lambd: rank regularization parameter
        - rho:   sparsity regularization parameter
        - tau:   proximal step size
        - niter: interations count for Douglas-Rachford
    Output:
        - Result of the Douglas-Rachford optimization with shape (n,n)
    """

    X0 = np.concatenate([X]*3)
    args = [X, lambd, rho, tau]
    XDR = DR(prox_denoise, prox_mean, theta, args, X0, niter=niter)
    A, B, C = np.array_split(XDR, 3)
    return (A+B+C)/3
