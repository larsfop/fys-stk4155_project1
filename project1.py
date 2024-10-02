import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import pandas as pd
from imageio import imread
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
import time

def FrankeFunction(x: npt.ArrayLike, y: npt.ArrayLike, noise_factor: int=0, rng: int = None) -> npt.NDArray:
    np.random.seed(rng)
    
    x, y = np.meshgrid(x, y)
    
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noise_factor*np.random.normal(0, 1, x.shape)

class LinearRegression:
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, z) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    def DesignMatrix(self, p: int, intercept: bool = True):
        if self.y.any() != None:
            self.X = np.ones((len(self.x), p + 1))
            for k in range(1, p + 1):
                j = k
                i = 0
                while i < k + 1:
                    self.X = np.column_stack((self.X, self.x**i * self.y**j))
                    j -= 1
                    i += 1
        
        else:
            self.X = np.ones((len(self.x), p + 1))
            for i in range(1, p + 1):
                self.X[:,i] = self.x**i
                
        if not intercept:
            self.X = np.delete(self.X, 0, 1)
            
    def Plot(self, beta) -> None:
        fig ,axs = plt.subplots(1, 2, sharex=True, sharey=True)
        
        axs[0].imshow(self.X @ beta, cmap=cm.coolwarm, origin='lower')
        axs[1].imshow(self.z, cmap=cm.coolwarm, origin='lower')
        axs[0].set_xlabel('X')
        axs[1].set_xlabel('X')
        axs[0].set_ylabel('Y')

        axs[0].set_title('Fitted data')
        axs[1].set_title('Real data')
        
    def PlotHeatMap(self, x, y, z, name: str):
        x, y = np.meshgrid(x, y)
        
        plt.figure()
        plt.yscale('log')
        cont = plt.contourf(x, y, z, locator=ticker.LogLocator())
        plt.colorbar(cont, aspect=5)
        
        plt.savefig(name+".pdf")
        
    def TrainAndTest(self, rng: int = None, lmbda: float = 0, bootstrap: int = 0) -> None:
        X_train, X_test, y_train, y_test = self.SplitData
        
        self.model.fit(X_train, y_train)
        beta = np.swapaxes(self.model.coef_, 0, 1)
        
        y_pred = self.model.predict(X_test) if bootstrap == 0 else np.empty((y_test.shape[0]*y_test.shape[1], bootstrap))
        for i in range(bootstrap):
            x, y = resample(X_train, y_train)
            
            y_pred[:, i] = self.model.fit(x, y).predict(X_test).ravel()
        
        if bootstrap != 0 :
            y_test = np.swapaxes(y_test.reshape(1, -1), 0, 1)
        error = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
        bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
        variance = np.mean(np.var(y_pred, axis=1, keepdims=True))
        
        # print(f'    Error = bias + variance = {bias:.4e} + {variance:.4e} = {error:.4e}')
        
        return error
    

class OrdinaryLeastSquare(LinearRegression):
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, z) -> None:
        super().__init__(x, y, z)
        self.model = linear_model.LinearRegression(fit_intercept=False)
        
class RidgeRegression(LinearRegression):
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, z, lambdas) -> None:
        super().__init__(x, y, z)
        self.model = linear_model.Ridge(fit_intercept=False)
        self.lambdas = lambdas
        
    def BootStrap(self, p, n_bootstrap, rng: int = None):
        MSE = np.zeros((len(self.lambdas), p + 1))
        for i in range(p + 1):
            self.DesignMatrix(i)
            self.SplitData = train_test_split(self.X, self.z, test_size=0.2, random_state=rng)
            for j in range(len(self.lambdas)):
                lmbda = self.lambdas[j]
                self.model.set_params(alpha=lmbda)
                # print(f'----------------------------------\nLambda = {lmbda:.2e}')
                MSE[j, i] = self.TrainAndTest(rng, lmbda, n_bootstrap)
                
        self.PlotHeatMap(np.arange(p+1), self.lambdas, MSE, 'RidgeBS_HM')
        
    def CrossValidation(self, p: int, k: int, rng: int = None, n_jobs: int = 1):
        param_grid = {
            'alpha': self.lambdas
        }
        clf = GridSearchCV(self.model, param_grid=param_grid, cv=k, scoring='neg_mean_squared_error', n_jobs=n_jobs)
        MSE = np.zeros((len(self.lambdas), p + 1))
        for i in range(p + 1):
            self.DesignMatrix(i)
            clf.fit(self.X, self.z)
            
            MSE[:,i] = -clf.cv_results_['mean_test_score']
        
        self.PlotHeatMap(np.arange(p + 1), self.lambdas, MSE, 'RidgeCV_HM')
        
        
class Lasso(LinearRegression):
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, z, lambdas) -> None:
        super().__init__(x, y, z)
        self.model = linear_model.Lasso(fit_intercept=False)
        self.lambdas = lambdas
        
        
if __name__ == "__main__":
    start = time.time()
    n = 100
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    
    z = FrankeFunction(x, y)
    
    lambdas = np.logspace(-8, 3, n)
    
    ridge = RidgeRegression(x, y, z, lambdas)
    # ridge.BootStrap(15, 0)
    ridge.CrossValidation(15, 5, n_jobs=8)
    
    end = time.time()
    print(f'Time: {end - start:.2f}s')
    
    plt.show()