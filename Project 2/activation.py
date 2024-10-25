import autograd.numpy as np
from autograd import elementwise_grad

def Sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))


def SoftMax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x)
    return exp/np.sum(exp, axis=1, keepdims=True)


def ReLU(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0)

def Linear(x: np.ndarray) -> np.ndarray:
    return x

