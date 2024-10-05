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
from sklearn.linear_model._base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
import time

def FrankeFunction(x: np.ndarray, y: np.ndarray, noise_factor: int=0, rng: int = None) -> np.ndarray:
    np.random.seed(rng)
    
    x, y = np.meshgrid(x, y)
    
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noise_factor*np.random.normal(0, 1, x.shape)

def R2(y_data: np.ndarray, y_model: np.ndarray) -> float:
    return np.mean(1 - np.sum((y_data - y_model) ** 2, axis=1, keepdims=True) / np.sum((y_data - np.mean(y_data)) ** 2))


def MSE(y_data: np.ndarray, y_model: np.ndarray) -> float:
    return np.mean(np.mean((y_data - y_model)**2, axis=1, keepdims=True))


class Estimator(BaseEstimator):
    def __init__(self, p: int = 1, lmbda: float = None, lin_reg = 'OLS'):
        """
            Custom estimator for GridSearchCV. Takes in lists or arrays for the polynomial order p and lambdas. Only works for X inputs where X is a 2d polynomial design matrix of order p.
            
            Parameters:
                p (int) : The order of polynomial for the design matrix.
                lmbda (float, optional) : The lambda value for Ridge and Lasso regression, default value is None. Required for Ridge and Lasso.
                lin_reg (str, linear_model) : String value for the implemented regression models (OLS, Ridge, Lasso), can also just be a regression model from sci-kit learn, though the estimator is quite simple and will probably not work with a different model.
        """
        self.p = p
        self.lmbda = lmbda
        
        if lin_reg == 'OLS':
            self.lin_reg = linear_model.LinearRegression(fit_intercept=False)
        elif lin_reg == 'Ridge':
            self.lin_reg = linear_model.Ridge(fit_intercept=False)
        elif lin_reg == 'Lasso':
            self.lin_reg = linear_model.Lasso(fit_intercept=False)
        else:
            self.lin_reg = lin_reg
            
    def fit(self, x: np.ndarray, y: np.ndarray):
        # Sets the lambda value if it is given as a parameter, if not default to what sci-kit learn have set
        if self.lmbda != None:
            self.lin_reg.set_params(alpha=self.lmbda)
        # an integer for picking the correct slice from the x values.
        n = int(((self.p + 2)*(self.p + 1))/2)
        x_ = x[:,:n]
        # print(self.p, n, x_.shape)
        self.lin_reg.fit(x_, y)
        
        return self
    
    def predict(self, x: np.ndarray):
        n = int(((self.p + 2)*(self.p + 1))/2)
        x_ = x[:,:n]
        return self.lin_reg.predict(x_)
    

class LinearRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, model: str) -> None:
        self.x = x
        self.y = y
        self.z = z

        self.model_key = model
        if model == 'OLS':
            self.model = linear_model.LinearRegression(fit_intercept=False)
        elif model == 'Ridge':
            self.model = linear_model.Ridge(fit_intercept=False)
        elif model == 'Lasso':
            self.model = linear_model.Lasso(fit_intercept=False)
        else:
            self.model = model

        self.params: dict[str, any] = {
            'p': [],
            'lmbda': []
        }
    
    def DesignMatrix(self, p: int, intercept: bool = True) -> np.ndarray:
        if self.y.any() != None:
            self.X = np.ones((len(self.x), 1))
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
            
        return self.X
    
    def SetParams(self, **params):
        self.params = params
            
    def Plot(self, beta: np.ndarray) -> None:
        fig ,axs = plt.subplots(1, 2, sharex=True, sharey=True)
        
        axs[0].imshow(self.X @ beta, cmap=cm.coolwarm, origin='lower')
        axs[1].imshow(self.z, cmap=cm.coolwarm, origin='lower')
        axs[0].set_xlabel('X')
        axs[1].set_xlabel('X')
        axs[0].set_ylabel('Y')

        axs[0].set_title('Fitted data')
        axs[1].set_title('Real data')
        
    def PlotHeatMap(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, name: str) -> None:
        x_, y_ = np.meshgrid(x, y)
        
        plt.figure()
        plt.yscale('log')
        cont = plt.contourf(x, y, z, norm='log')
        # cont = plt.imshow(np.log(z), cmap=cm.coolwarm, origin='lower', extent=(x[0], x[-1], y[0], y[-1]))
        plt.colorbar(cont, aspect=5)
        
        plt.savefig(name+".pdf")

    def TrainAndTest(self, *params, p: int = 5, lmbda: float = None, rng: int = None, bootstrap: int = 0) -> None:
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.z, test_size=0.2, random_state=rng)

        if lmbda != None:
            self.model.set_params(alpha=lmbda)

        if len(params) == 1:
            key = params[0]
            param = self.params[params[0]]
            MSE_test = np.zeros(len(param))
            MSE_Train = np.zeros(len(param))

            for i in range(len(param)):
                X_train_, X_test_ = X_train, X_test
                if key == 'p':
                    n = int(((i + 2)*(i + 1))/2)
                    X_train_, X_test_ = X_train[:,:n], X_test[:,:n]
                elif key == 'lmbda':
                    self.model.set_params(alpha=param[i])

                self.model.fit(X_train_, y_train)

                y_fit = self.model.predict(X_train_)
                y_pred = self.model.predict(X_test) if bootstrap == 0 else np.empty((y_test.shape[0]*y_test.shape[1], bootstrap))
                for k in range(bootstrap):
                    x, y = resample(X_train, y_train)
                    
                    y_pred[:, k] = self.model.fit(x, y).predict(X_test).ravel()
                
                if bootstrap != 0 :
                    y_test = np.swapaxes(y_test.reshape(1, -1), 0, 1)

                MSE_Train[i] = MSE(y_train, y_fit)
                MSE_test[i] = MSE(y_test, y_pred)

                # print(f'{'MSE:' : <4} {mean_squared_error(y_test, y_pred):g}')
                # print(f'{'R2:' : <4} {r2_score(y_test, y_pred):g}')

            plt.figure()
            plt.grid()
            if key == 'lmbda':
                plt.xscale('log')
            plt.plot(param, MSE_test, label='Test Set')
            plt.plot(param, MSE_Train, label='Train set')
            plt.legend()

        elif len(params) == 2:
            k1, k2 = params[0], params[1]
            p1, p2 = self.params[k1], self.params[k2]

            MSE_test = np.zeros( ( (
                len(p1), 
                len(p2)
                ) ) )
            
            plt.figure()
            plt.grid()
            plt.xscale('log')

            for i in range(len(p1)):
                X_train_, X_test_ = X_train, X_test
                if k1 == 'p':
                    n = int(((i + 2)*(i + 1))/2)
                    X_train_, X_test_ = X_train[:,:n], X_test[:,:n]
                elif k1 == 'lmbda':
                    self.model.set_params(alpha=p1[i])

                for j in range(len(p2)):
                    if k2 == 'p':
                        n = int(((j + 2)*(j + 1))/2)
                        X_train_, X_test_ = X_train[:,:n], X_test[:,:n]
                    elif k2 == 'lmbda':
                        self.model.set_params(alpha=p2[j])

                    self.model.fit(X_train_, y_train)

                    y_fit = self.model.predict(X_train_)
                    y_pred = self.model.predict(X_test) if bootstrap == 0 else np.empty((y_test.shape[0]*y_test.shape[1], bootstrap))
                    for k in range(bootstrap):
                        x, y = resample(X_train, y_train)
                        
                        y_pred[:, k] = self.model.fit(x, y).predict(X_test).ravel()
            
                    if bootstrap != 0 :
                        y_test = np.swapaxes(y_test.reshape(1, -1), 0, 1)

                    # MSE_Train[i] = mean_squared_error(y_train, y_fit)
                    MSE_test[i, j] = MSE(y_test, y_pred)

                    # print(f'{'MSE:' : <4} {mean_squared_error(y_test, y_pred):g}')
                    # print(f'{'R2:' : <4} {r2_score(y_test, y_pred):g}')

                plt.plot(p2, MSE_test[i,:], label=f'p = {p1[i]}')

            plt.legend()
            
        else:
            self.model.fit(X_train, y_train)

            y_fit = self.model.predict(X_train)
            y_pred = self.model.predict(X_test) if bootstrap == 0 else np.empty((y_test.shape[0]*y_test.shape[1], bootstrap))
            for k in range(bootstrap):
                x, y = resample(X_train, y_train)
                
                y_pred[:, k] = self.model.fit(x, y).predict(X_test).ravel()
            
            if bootstrap != 0 :
                y_test = np.swapaxes(y_test.reshape(1, -1), 0, 1)

            print(f'{'MSE:' : <4} {MSE(y_test, y_pred):g}')
            print(f'{'R2:' : <4} {R2(y_test, y_pred):g}')

    def CrossValidation(self, *params, name: str, p_order: int = 5, lmbda: float = None, k: int = 10, n_jobs: int = 1) -> None:
        """
            This function can do a Cross Validation for Linear Regression, with the current models, (Ordinary Least Square, Ridge, Lasso). 
            Additionally it can perform a grid search over the following parameters (lambda, p). Read Parameters for further information.

            Parameters:
                *params : Parameters to be set for the Grid Search, available is 'lmbda' and 'p'. Leave empty for Cross validation without Grid Search.
                name (str) : Name for saving plots, will save as <name>_<type>.pdf, where <type> is dependent on what plots are generated.
                p_order (int, optional) : Polynomial order for when it is not given as a parameter for Grid Search. Default is 5th order.
                lmbda (float, optional) : Lambda parameter for Ridge and Lasso regression, default is None due to Ordinary Least Square not requiring the parameter.
                k (int, optional) : Number of folds for Cross Validation, default is 10.
                n_jobs (int, optional) : Set the number of threads that sci-kit can use in Grid Search and Cross Validation. Default is 1 and set to -1 to use every thread available.
        """
        param_grid = {}
        for param in params:
            param_grid[param] = self.params[param]

        cv = GridSearchCV(Estimator(lin_reg=self.model_key, p=p_order, lmbda=lmbda),
                      cv=k,
                      param_grid=param_grid,
                      scoring='neg_mean_squared_error',
                      n_jobs=n_jobs)
    
        cv.fit(self.X, self.z)

        """print(cv.best_index_)
        df = pd.DataFrame(cv.cv_results_)
        MSE_std = df['rank_test_score' == cv.best_index_]
        print(pd.DataFrame(MSE_std))"""
        
        print(f'{'Parameter' : <12} Value')
        print('---------------------------')
        for param, value in cv.best_params_.items():
            print(f'{'Best '+param : <12} {value:g}')
        print(f'{'Best MSE' : <12} {-cv.best_score_:.5f}')
        print(f'{'CV score': <12} {-np.mean(cross_val_score(cv.best_estimator_, self.X, self.z, scoring="neg_mean_squared_error", cv=k)):.5f}')

        n = len(params)
        results = -cv.cv_results_['mean_test_score']
        if n == 1:
            plt.figure()
            MSE = results
            plt.plot(param_grid[params[0]], MSE)
        elif n > 1:
            p = len(param_grid['p'])
            l = len(param_grid['lmbda'])
            MSE = np.zeros((l, p))

            plt.figure()
            plt.xscale('log')
            for i in range(p):
                MSE[:, i] = results[i::p]
                plt.plot(param_grid['lmbda'], results[i::p], label=f'p = {i}')

            plt.legend()

            self.PlotHeatMap(param_grid['p'], param_grid['lmbda'], MSE, name+'_HM')

        
if __name__ == "__main__":
    start = time.time()
    n = 100
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
        
    z = FrankeFunction(x, y, 0)
    
    p = 5
    
    lambdas = np.logspace(-8, 1, n)
    ps = np.arange(p+1)
    
    k = 10
    ridge = LinearRegression(x, y, z, 'Ridge')
    ridge.DesignMatrix(p)
    ridge.SetParams(p=ps, lmbda=lambdas)
    ridge.TrainAndTest('lmbda', bootstrap=20)
    # ridge.CrossValidation('p', 'lmbda', name='Ridge_CV' ,n_jobs=8)

    # ols = LinearRegression(x, y, z, 'OLS')
    # ols.DesignMatrix(p)
    # ols.SetParams(p=ps)
    # ols.TrainAndTest('p')
    # ols.CrossValidation('p', name='OLS_CV', n_jobs=8)


    end = time.time()
    print(f'Time: {end - start:.2f}s')
    
    plt.show()