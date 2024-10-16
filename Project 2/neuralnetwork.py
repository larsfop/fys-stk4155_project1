import autograd.numpy as no
from autograd import grad
from sklearn.model_selection import train_test_split

import optim as opt
import loss_fn as lf
from activation import *

dict(
    input = 8,
    layer1 = [12, Sigmoid],
    layer2 = [20, Sigmoid],
    ouput = [3, SoftMax]
)

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
        X_data: np.ndarray,
        y_data: np.ndarray,
        optimizer: opt.BaseOptimizer,
        loss_fn: lf.BaseLossFunction,
        learning_rate: float,
        model_structure: dict,
        lr: float = 1e-4,
        **hyper_params,
        ) -> None:
        
        self.X = X_data
        self.y = y_data
        
        self.model_structure = model_structure
        self.grad_activation_func = []
        for values in model_structure.values():
            if len(values) == 2:
                self.grad_activation_func.append(grad(values['activation'], 0))
        
        self.eta = lr
        
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def split_data(self, split: float = 0.2, rng: int = None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=split, random_state=rng)
        
    def create_layers(self):
        self.layers = []
        
        i_size = self.model_structure['input']
        for key, output_size in self.model_structure.items():
            if key != 'input':
                W = np.random.randn(i_size, output_size)
                b = np.random.randn(output_size)
                self.layers.append((W, b))

                i_size = output_size
        
    def feed_forward(self, X):
        self.probabilities = []
        self.a = []
        self.z = []
        
        a = X
        self.a.append(a)
        # self.z.append(a)
        self.probabilities.append(self.X_train)
        for (W, b), activation in zip(self.layers, self.activations):
            z = a @ W + b
            a = activation(z)
            
            self.a.append(a)
            self.z.append(z)
            self.probabilities.append(a)
        
    def back_propagation(self):
        error_output = self.probabilities[-1] - self.y_train
        activation_func_derivative = reversed(self.grad_activation_func)
        
        # Compute the gradients in the hidden layers
        error = error_output
        W, b = self.layers[-1]
        #for (W, b), a, z in zip(reversed(self.layers[:-1]), reversed(self.a[:-1]), reversed(self.z[-1])):
        for i in range(len(self.layers) - 1, -1, -1): 
            ## Output Layer ##
            if i == len(self.layers) - 1:
                a = self.a[i+1]
                error = a - self.y
                
                a = self.a[i]
                W, b = self.layers[i]
                W -= self.eta * a.T @ error
                
                # Regularization term
                W += self.lmbda * W
                b -= np.sum(error, axis=0)
                
                self.layers[i] = (W, b)
            else:
                a = self.a[i]
                z = self.z[i]
                error = error @ W.T * activation_func_derivative[i](z)
                
                W, b = self.layers[i]
                # Compute new weight
                W -= self.eta * a.T @ error
                # Regularization term
                W += self.lmbda * W
                # Compute new bias
                b -= self.eta * np.sum(error, axis=0)
                
                self.layers[i] = (W, b)
            
    def train(self, epochs):
        for i in range(epochs):
            self.feed_forward()
            self.back_propagation()
    
if __name__=="__main__":
    d = dict(
        input = 8,
        layer1 = (12, Sigmoid),
        ouput = (3, SoftMax),
    )
    a = dict(
        input = {"num": 8},
        hidden_layer1 = {"num": 12, 'activation': Sigmoid},
        output = {"num": 3, "activation": SoftMax}
    )
    for val in a.values():
        if len(val) == 2:
            print(val['activation'].__name__)
    
    create_layers(d)
    
    x = [(1,2), (3,4), (4,5)]
        
    for (a,b), in zip(reversed(x)):
        print(a, b)