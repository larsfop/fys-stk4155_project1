import autograd.numpy as np
from autograd import elementwise_grad

def DesignMatrix(p_order: int, *arrays: np.ndarray, intercept: bool = True) -> np.ndarray:
    """
    Produce a Design matrix for linear regression for a given polynomial order in 1d or omit y for 2d

    Parameters:
        p_order (int): Order of polynomial
        arrays (ArrayLike): Arrays to be used in the Design matrix, input one array for 1d and two arrays for 2d.
        intercept (bool, optional): Wether the intercept should be included in the Design matrix, if False remove the first parameter
    """
    if len(arrays) > 2:
        raise ValueError(f"Number of inputed arrays are too large, at most two arrays can be inputeted not {len(arrays)}.")

    if len(arrays) == 2:
        x, y = arrays
        X = np.ones((len(x), 1))
        for k in range(1, p_order+1):
            j = k
            i = 0
            while i < k + 1:
                X = np.column_stack((X, x**i*y**j))
                j -= 1
                i += 1
    
    else:
        x = arrays[0]
        X = np.ones((len(x), p_order+1))
        for i in range(1, p_order+1):
            X[:,i] = x**i

    if not intercept:
        X = np.delete(X, 0, 1)

    return X


class ModelDict:
    def __init__(self, input_size: int, layers_shape: list, activations: list) -> None:
        self.model_dict = {'input': {'num': input_size}}
        i = 1
        for layer, activation in zip(layers_shape, activations):
            self.model_dict[f'layer_{i}'] = {'num': layer, 'activation': activation}
            i += 1
            
    def __str__(self) -> str:
        string = "ModelDict:\n"
        i = 1
        for key, value in self.model_dict.items():
            if key == 'input':
                string += f"{'Input:': >9} {'Size =': >8} {value['num']}\n"
            else:
                string += f"{'Layer': >8} {i}: Size = {value['num']}, Activation = {value['activation'].__name__}\n"
                i += 1
                
        return string
    
    
def derivative(func):
    if func.__name__ == "ReLU":
        def func(x):
            return np.where(x > 0, 1, 0)
        
        return func
        
    else:
        return elementwise_grad(func)