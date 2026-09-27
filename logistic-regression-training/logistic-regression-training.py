import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    
    def get_predictions():
        return _sigmoid(X @ weights + bias)

    def cross_entropy_loss():
        y_hat = get_predictions()
        return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))/np.size(y, 0)
    def compute_weight_gradients():
        return X.T @ (get_predictions() - y) /np.size(y, 0)
    def compute_bias_gradient():
        return np.sum(get_predictions() - y)/np.size(y, 0)
        

    
    # Write code here
    weights = np.zeros(np.size(X, 1))
    bias = 0

    for step in range(steps):
        # print(f"Initial Weights: {weights}, Bias: {bias}")
        # print(f"Current predictions: {get_predictions()}")
        # print(f"Current loss: {cross_entropy_loss()}")
    
        # print(f"Weight gradient: {compute_weight_gradients()}")
        # print(f"Bias gradient: {compute_bias_gradient()}")
        weight_grads = (lr * compute_weight_gradients())
        bias_grad = (lr * compute_bias_gradient())
        weights -= weight_grads
        bias -= bias_grad

    print(f"Final weights: {weights}")
    print(f"Final bias: {bias}")
    print(f"Final loss: {cross_entropy_loss()}")
    return (weights, bias)