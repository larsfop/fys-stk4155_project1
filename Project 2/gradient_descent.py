
import numpy as np
from sklearn.metrics import root_mean_squared_error
from typing import Callable
from optim import BaseOptimizer
from loss_fn import BaseLossFunction

class GradientDescent:
    def __init__(self, 
                 X_train: np.ndarray, 
                 X_test: np.ndarray, 
                 y_train: np.ndarray, 
                 y_test: np.ndarray, 
                 loss_function: BaseLossFunction,
                 lmbda: float = 0.1,
                 optimizer: BaseOptimizer = None, 
                 lr: float = 1e-2, 
                 decay: float = 0,
                 momentum: float = 0, 
                 nestrov: bool = False,
                 batch_size: int = -1,
                 rng: int = None,
                 verbose: int = 1,
                 **optim_params
        ) -> None:

        """
        Implementation of Gradient Descent.
        
        Parameters:
            X (ndarray): A numpy array containing the features, must have shape (n, p), where n is number of datapoints and p is the order of polynomial.
            y (ndarray): A numpy array containing the true data points of shape (n).
            optimizer (optional): Selects the optimizer for the gradient descent, leave empty for a standard gradient descent.
            lr (float): Learning rate.
            decay (float): The decay constant for the learning rate.
            momentum (float): Momentum constant for gradient descent with momentum.
            nestrov (bool) : Use Nestrov momentum instead of standard momentum.
            batch_size (int): The batch size to use for stochastic gradient descent.
            optim_params : Additional keyword arguments for the optimizer you want to use.
        """
        
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.nestrov = nestrov
        
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        
        self.n, self.p = self.X_train.shape
        
        self.M = self.n if batch_size == -1 else batch_size
        self.m = int(self.n/self.M)
        
        if isinstance(optimizer, Callable):
            self.optimizer: BaseOptimizer = optimizer(lr=lr, verbose=verbose, batch_size=batch_size, **optim_params)
        elif optimizer == None:
            self.optimizer = BaseOptimizer(lr=lr)
        else:
            raise ValueError(f"The followin arguement is not valid as optimizer, use default or one of the available optimizers:\n \
                             ({'; '.join([type(c()).__name__ for c in BaseOptimizer.__subclasses__()])})")
            
        if isinstance(loss_function, Callable):
            self.loss_fn: BaseLossFunction = loss_function(lmbda=lmbda)
        else:
            self.loss_fn = loss_function
            
    def SetOptimizer(self, optimizer, **optim_params) -> None:
        if isinstance(optimizer, Callable):
            self.optimizer: BaseOptimizer = optimizer(lr=self.lr, batch_size=self.M, **optim_params)
        else:
            self.optimizer = BaseOptimizer(lr=self.lr)
        
    def FindOptimalTheta(self, theta: np.ndarray = None, epochs: int = 1e3, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = np.random.uniform(0, 1, self.p) if theta is None else theta
        lr = self.lr
        
        MSE_train = np.zeros(epochs)
        MSE_test = np.zeros(epochs)
        
        change = 0
        epoch = 0
        while epoch < epochs:# and error > tol:
            for i in range(self.m):
                k = self.M * np.random.randint(self.m)
                Xi = self.X_train[k:k+self.M, :]
                yi = self.y_train[k:k+self.M]
                
                theta += self.momentum*change * self.nestrov
                
                # gradients = 2.0/self.M * Xi.T @ ((Xi @ theta) - yi)
                gradients = self.loss_fn.Gradients(X=Xi, y=yi, theta=theta, n=self.M)
                
                change = self.optimizer.dTheta(gradients) + self.momentum*change
                
                theta += change
                
            y_fit = self.X_train @ theta
            y_pred = self.X_test @ theta
            
            MSE_train[epoch] = root_mean_squared_error(self.y_train, y_fit)**2
            MSE_test[epoch] = root_mean_squared_error(self.y_test, y_pred)**2
            
            lr *= 1.0/(1 + self.decay * epoch)
            self.optimizer.SetLearningRate(lr)
            epoch += 1
        # MSE_train /= self.m
        # MSE_test /= self.m

        return theta, MSE_train, MSE_test