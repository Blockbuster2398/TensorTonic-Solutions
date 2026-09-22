def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    
    def q_derivative(x : float) -> float:
        return (2 * a * x) + (b) + 0

    for i in range(steps):
        x0 = x0 + (lr * -q_derivative(x0))
    
    # Write code here
    return x0