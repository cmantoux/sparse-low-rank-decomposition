"""
This file implements the proximal operators used throughout the rest of the code.
"""

import numpy as np


def soft_threshold(A, t):
    """
    Soft thresholding operator, as defined in the paper.
    """
    B = np.maximum(np.abs(A)-t, 0)
    return np.sign(A)*B

def prox_norm_1(A, t, offset=None):
    """
    Proximal operator for the L1 norm.
    """
    if offset is None:
        return soft_threshold(A, t)
    else:
        return offset + prox_norm_1(A-offset, t)

def prox_norm_nuclear(A, t, offset=None):
    """
    Proximal operator for the nuclear norm.
    """
    if offset is None:
        u, s, vh = np.linalg.svd(A)
        s2 = soft_threshold(s, t)
        return u@np.diag(s2)@vh
    else:
        return offset + prox_norm_nuclear(A-offset, t)
