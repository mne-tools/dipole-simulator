import numpy as np


def find_closest(a, x):
    """Find the element in the array a that's closest to the scalar x.
    """
    a = a.squeeze()
    idx = np.abs(a - x).argmin()
    return a[idx]
