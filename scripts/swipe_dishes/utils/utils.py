import numpy as np

def angle_diff(a:float, b:float)->float:
    """calculate minumum angle difference

    Args:
        a (float): angle 1
        b (float): angle 2

    Returns:
        float: angle difference between two angles
    """
    if np.sign(a * b) < 0:
        if np.abs(b - a) < np.abs(b - a + np.sign(a) * 2 * np.pi):
            return b - a
        else:
            return b - a + np.sign(a) * 2 * np.pi
    else:
        return b - a
    