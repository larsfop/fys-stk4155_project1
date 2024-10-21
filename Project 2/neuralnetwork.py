import autograd.numpy as np
from autograd import grad, elementwise_grad
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import optim as opt
import loss_fn as lf
from activation import *
from utils import DesignMatrix, ModelDict

def create_layers(model_structure: dict):
    layers = []
        
    i_size = model_structure['input']
    for key, value in model_structure.items():
        if key != "input":
            print(key, value)
            output_size = value[0]
            W = np.random.randn(i_size, output_size)
            b = np.random.randn(output_size)
            layers.append((W, b))

            i_size = output_size
    for w, b in layers:
        print(w.shape, b.shape)

class NeuralNetwork:
    def __init__(
        self,
        model_dict: ModelDict,
        optimizer: opt.BaseOptimizer = opt.ADAM,
        loss_fn: lf.BaseLossFunction = lf.OLS,
        learning_rate: float = 1e-3,
        regulatization = 0,
        **hyper_params,
        ) -> None:
        
        self.model_structure = model_dict.model_dict
        self.grad_activation_func = []
        self.activations = []
        for values in self.model_structure.values():
            if len(values) == 2:
                self.activations.append(values['activation'])
                self.grad_activation_func.append(derivative(values['activation']))
        
        self.eta = learning_rate
        self.regularization = regulatization
        
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.derivative_loss_fn = grad(loss_fn)
        
    # def split_data(self, X: np.ndarray, y: np.ndarray, split: float = 0.2, rng: int = None):
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=split, random_state=rng)
        
    def create_layers(self):
        self.layers = []
        
        i_size = self.model_structure['input']['num']
        for key, output_size in self.model_structure.items():
            if key != 'input':
                output_size = output_size['num']
                W = np.random.randn(i_size, output_size)
                b = np.random.randn(output_size)
                self.layers.append((W, b))

                i_size = output_size
        
    def feed_forward(self, X: np.ndarray):
        self.a = []
        self.z = []
        
        a = X
        self.a.append(a)
        # self.z.append(a)
        for (W, b), activation in zip(self.layers, self.activations):
            z = a @ W + b
            a = activation(z)
            
            self.a.append(a)
            self.z.append(z)
            
        return a
        
    def back_propagation(self, X: np.ndarray, t: np.ndarray):
        # error_output = self.a[-1] - self.y_train
        activation_func_derivative = list(reversed(self.grad_activation_func))
        
        # Compute the gradients in the hidden layers
        # error = error_output
        W, b = self.layers[-1]
        #for (W, b), a, z in zip(reversed(self.layers[:-1]), reversed(self.a[:-1]), reversed(self.z[-1])):
        for i in range(len(self.layers) - 1, -1, -1): 
            ## Output Layer ##
            if i == len(self.layers) - 1:
                a = self.a[i+1]
                # error = a - self.y_train
                
                z = self.z[i]
                error = activation_func_derivative[i](z) * grad(self.loss_fn(t))(a)

                a = self.a[i]
                W, b = self.layers[i]
                W -= self.eta * (a.T @ error + self.regularization * W)
                
                b -= self.eta * np.sum(error, axis=0)
                
                self.layers[i] = (W, b)
            else:
                a = self.a[i]
                z = self.z[i]
                error = error @ W.T * activation_func_derivative[i](z)
                
                W, b = self.layers[i]
                # Compute new weight
                W -= self.eta * (a.T @ error + self.regularization * W)
                # Compute new bias
                b -= self.eta * np.sum(error, axis=0)
                
                self.layers[i] = (W, b)
            
    def train(self, X: np.ndarray, t: np.ndarray, epochs: int = 1000):
        for i in range(epochs):
            self.feed_forward(X)
            self.back_propagation(X, t)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predict = self.feed_forward(X)
        
        return predict


def OLS(target):
    
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func

    
if __name__=="__main__":
    np.random.seed(125)

    n = 100
    x = np.sort(np.random.uniform(-1, 1, n))
    y = 2*x + 9*x**2 + 4*x**3
    
    y = y.reshape(-1, 1)
    
    # print(f'{1:06.2f}')
    
    p = 3
    X = DesignMatrix(p, x)
    # print(X)
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X, y)
    # print(ols.coef_)
    # print(ols.predict(X))
    # print(y)
    
    md = ModelDict(p+1,
                         [12, 10, 1],
                         [Sigmoid, Sigmoid, ReLU]
                    )
    # print(md)
    
    test = NeuralNetwork(md, learning_rate=1e-1, loss_fn=OLS)
    test.split_data(X, y)
    test.create_layers()
    test.train()
    
    pred = test.predict(X)
    # print(pred)
    # print(pred.shape)

    plt.scatter(x, y)
    plt.scatter(x, pred)

    plt.show()