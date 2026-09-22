import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    param = np.asarray(param, dtype="float64")
    print(f"Current parameters: {param}")
    grad = np.asarray(grad)
    print(f"Gradients for each parameter: {grad}")
    m = np.asarray(m)
    print(f"Starting m for each parameter: {m}")
    v = np.asarray(v)
    print(f"Starting v for each parameter: {v}")
    
    print(f"Starting t: {t}")
    

    
    # Update first moment:
    m = (beta1 * m) + (1-beta1) * grad
    # Update second moment:
    v = (beta2 * v) + (1-beta2) * (grad ** 2)

    # Bias correction:
    m_hat = m/(1-beta1**(t))
    v_hat = v/(1-beta2**(t))

    param = param -  (lr * m_hat/(np.sqrt(v_hat) + eps))
    
    return (param, m, v)