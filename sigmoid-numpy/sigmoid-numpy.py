import numpy as np
import math
def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    if isinstance(x, list):
        return 1/(1 + (math.e ** (-np.asarray(x))))
    elif isinstance(x, (float, int)):
        return 1/(1 + (math.e ** (-x)))
    pass