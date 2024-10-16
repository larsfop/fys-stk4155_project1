import numpy as np

class BaseOptimizer:
    def __init__(self, lr: float = 1e-2, verbose: int = 1, **_) -> None:
        self.lr = lr
        self.params = {}
        
        if len(_) > 0 and verbose == 1:
            print('### Warning! Unused Parameters ###')
            print(f"With opimizer '{type(self).__name__}' the following parameters are unused")
            for key, value in _.items():
                print(f"    Parameter '{key}' with value '{value}'.")
        
    def dTheta(self, gradients: np.ndarray) -> np.ndarray:
        return -self.lr * gradients
    
    def SetParameter(self, **params) -> None:
        for key, value in params.items():
            self.params[key] = value
            
    def SetLearningRate(self, lr) -> None:
        self.lr = lr
    
        
class LrScaler(BaseOptimizer):
    def __init__(self, lr: float = 0.01, batch_size: int = 2, verbose: int = 1, **_) -> None:
        super().__init__(lr, verbose=verbose, **_)
        
        self.lr = lr*batch_size/128
        
    def dTheta(self, gradients):
        return -self.lr * gradients
    
    
class LrSqrtScaler(BaseOptimizer):
    def __init__(self, lr: float = 0.01, batch_size: int = 2, verbose: int = 1, **_) -> None:
        super().__init__(lr, verbose=verbose, **_)
        
        self.lr = lr*np.sqrt(batch_size)/np.sqrt(128)
        

class AdaGrad(BaseOptimizer):
    def __init__(self, lr: float = 1e-2, delta: float = 1e-6, verbose: int = 1, **_) -> None:
        super().__init__(lr, verbose=verbose, **_)

        self.lr = lr
        self.delta = delta
        self.r = 0
        
    def dTheta(self, gradients: np.ndarray) -> np.ndarray:
        self.r += gradients**2
        return -self.lr/(self.delta + np.sqrt(self.r)) * gradients
    
    
class RMSProp(BaseOptimizer):
    def __init__(self, lr: float = 1e-2, delta: float = 1e-6, rho: float = 0.99, verbose: int = 1, **_) -> None:
        super().__init__(lr, verbose=verbose, **_)
        
        self.lr = lr
        self.delta = delta
        self.rho = rho
        self.r = 0
        
    def dTheta(self, gradients: np.ndarray) -> np.ndarray:
        self.r = self.rho*self.r + (1 - self.rho) * gradients**2
        return -self.lr/np.sqrt(self.delta + self.r) * gradients
    
    
class ADAM(BaseOptimizer):
    def __init__(self, lr: float = 1e-3, delta: float = 1e-8, rho1: float = 0.9, rho2: float = 0.999, verbose: int = 1, **_) -> None:
        super().__init__(lr, verbose=verbose, **_)
        
        self.lr = lr
        self.delta = delta
        self.rho1 = rho1
        self.rho2 = rho2
        
        self.s = 0
        self.r = 0
        self.t = 0
        
    def dTheta(self, gradients: np.ndarray) -> np.ndarray:
        self.t += 1
        
        self.s = self.rho1*self.s + (1 - self.rho1) * gradients
        s_hat = self.s/(1 - self.rho1**self.t)
        
        self.r = self.rho2*self.r + (1 - self.rho2) * gradients**2
        r_hat = self.r/(1 - self.rho2**self.t)
        
        return -self.lr*s_hat/(np.sqrt(r_hat) + self.delta)