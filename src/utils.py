import numpy as np
import matplotlib.pyplot as plt


norm = np.linalg.norm


class Counter:
    """
    Class to track the iteration count inside the main loop skopt.forest_minimize.
    """
    def __init__(self):
        self.k = 0
    def increment(self):
        self.k += 1


def sparsity(A, tolerance=0):
    """
    Returns the proportion of null coefficients in A, up to a tolerance threshold.
    """
    if tolerance==0:
        return (A==0).mean()
    else:
        return (np.abs(A)<=tolerance).mean()


def relative_error(A, B, order=None, rnd=20):
    """
    Computes the relative error between A and B, rounded to the desired precition.
    The default is the L2 norm, i.e. the relative Root Mean Square Error.
    """
    return (norm(A-B, order)/norm(B, order)).round(rnd)