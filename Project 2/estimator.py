
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from neuralnetwork import NeuralNetwork, OLS
import optim
import loss_fn
import activation
from utils import ModelDict, derivative, DesignMatrix
import autograd.numpy as np
from autograd import grad



class NNEstimator(BaseEstimator, NeuralNetwork):
    def __init__(
        self,
        model_structure: ModelDict,
        optimizer: optim.BaseOptimizer = optim.ADAM,
        loss_fn: loss_fn.BaseLossFunction = loss_fn.OLS,
        eta: float = 1e-3,
        epochs: int = 1000,
        regularization: float = 0,
    ):
        self.model_structure = model_structure
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        # Declare all valid hyperparameters for grid search
        self.eta = eta
        self.epochs = epochs
        self.regularization = regularization
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            assert hasattr(self, parameter), f"{parameter} is not a valid hyperparameter"
            setattr(self, parameter, value)
            
        return self
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model_structure = self.model_structure.model_dict
        
        self.grad_activation_func = []
        self.activations = []
        for values in self.model_structure.values():
            if len(values) == 2:
                self.activations.append(values['activation'])
                self.grad_activation_func.append(derivative(values['activation']))
                
        self.derivative_loss_fn = grad(self.loss_fn)
        
        self.create_layers()
        
        self.train(X, y, self.epochs)
        
        return self
        
    def predict(self, X: np.ndarray):
        
        return self.nn_predict(X)
    
    
if __name__=="__main__":
    
    n = 100
    x = np.linspace(-1, 1, n)
    y = 2*x + 9*x**2 + 4*x**3
    
    p = 3
    X = DesignMatrix(p, x)
    
    md = ModelDict(
        p+1,
        [10, 1],
        [activation.Sigmoid, activation.ReLU]
    )
    
    param_grid = {'eta': np.logspace(-6, -1, 10)}
    cv = GridSearchCV(
        estimator=NNEstimator(
            model_structure=md,
            loss_fn=OLS,
            epochs=100
        ),
        param_grid=param_grid,
        cv=5,
        error_score='raise',
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    
    cv.fit(X, y)