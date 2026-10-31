import numpy as np

def create_sequence(start: float, stop: float, param: float, kind: str) -> np.ndarray:
    """
    Returns a 1D float64 array containing the requested sequence.
    """
    return np.arange(start, stop, param, dtype=np.float64) if kind == "arange" else np.linspace(start, stop, param, endpoint=True, dtype=np.float64)
