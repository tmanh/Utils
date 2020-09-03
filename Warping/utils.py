import numpy as np
from numba import jit


@jit('int32(float32)')
def round(x):
    if x - np.int32(x) > 0.5:
        return np.int32(x) + 1
    return np.int32(x)
