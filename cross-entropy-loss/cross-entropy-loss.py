import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    epsilon = .0000001
    y_pred = np.clip(np.asarray(y_pred), epsilon, 1-epsilon)
    y_true = np.asarray(y_true)
    
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")

    correctPredictionProbs = y_pred[np.arange(0, y_true.size), y_true]
    print(f"correctPP: {correctPredictionProbs}")

    return np.average(np.log(correctPredictionProbs)) * -1
    
    pass