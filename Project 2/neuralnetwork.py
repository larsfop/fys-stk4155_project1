import autograd.numpy as np
from autograd import grad, elementwise_grad
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import matplotlib.pyplot as plt

import optim as opt
import loss_fn as lf
from activation import *
from utils import DesignMatrix, ModelDict, derivative

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
    
    def nn_predict(self, X: np.ndarray) -> np.ndarray:
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
    
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # y = scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    scaler.fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)

    from sklearn.linear_model import LinearRegression
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_train, y_train)
    
    
    md = ModelDict(
        p+1,
        [100, 1],
        [Sigmoid, Linear]
    )
    # print(md)
    
    test = NeuralNetwork(md, learning_rate=1e-1, loss_fn=OLS, regulatization=0)
    test.create_layers()
    test.train(X_train, y_train)
    
    pred = test.nn_predict(X_test)
    
    x = X_test[:,1]
    y_reg = ols.predict(X_test)
    y_reg = y_reg[x.argsort()]
    
    y_nn = test.nn_predict(X_test)
    y_nn = y_nn[x.argsort()]
    
    x = np.sort(x)
    
    plt.plot(x, y_reg, label='Linear Regression')
    plt.plot(x, y_nn, '--', label='Neural Network')

    plt.grid()
    plt.legend()
    plt.show()