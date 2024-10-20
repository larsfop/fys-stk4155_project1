import autograd.numpy as np

class BaseLossFunction:
    def __init__(self) -> None:
        pass
    
    def Gradients(self) -> np.ndarray:
        pass


class OLS(BaseLossFunction):
    def __init__(self, **_) -> None:
        pass

    def Beta(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(X.T @ X) @ X.T @ y
    
    def Gradients(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray, n: int) -> np.ndarray:
        return 2.0/n * X.T @ (X @ theta - y)
        
        
class Ridge(BaseLossFunction):
    def __init__(self, lmbda: float = 0.1, **_) -> None:
        self.lmbda = lmbda

    def Beta(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        I = np.eye(X.shape[1], X.shape[1])
        return np.linalg.inv(X.T @ X + self.lmbda*I) @ X.T @ y
        
    def Gradients(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray, n: int) -> np.ndarray:
        return 2.0/n * X.T @ (X @ theta - y) + 2*self.lmbda*theta
    