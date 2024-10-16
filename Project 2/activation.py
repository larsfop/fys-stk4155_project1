import numpy as np

def Sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))

def SoftMax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x)
    return exp/np.sum(exp, axis=1, keepdims=True)