import numpy as np
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
from matplotlib.colors import LogNorm
from random import random, seed
import time
import seaborn as sns
from typing import Tuple, Callable

def FrankeFunction(x: np.ndarray, y: np.ndarray, noise: float = 0, rng: int = None) -> np.ndarray:
    np.random.seed(rng)
    
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0, noise, x.shape)

def R2(y_data: np.ndarray, y_model: np.ndarray) -> float:
    return np.mean(1 - np.sum((y_data - y_model) ** 2, axis=1, keepdims=True) / np.sum((y_data - np.mean(y_data)) ** 2))


def MSE(y_data: np.ndarray, y_model: np.ndarray) -> float:
    return np.mean(np.mean((y_data - y_model)**2, axis=1, keepdims=True))


def Bias(y_data: np.ndarray, y_model: np.ndarray) -> float:
    y_model = y_model.reshape(-1, 1)
    return np.mean( (y_data - np.mean(y_model, axis=1, keepdims=True))**2 )


def Variance(y_model: np.ndarray) -> float:
    y_model = y_model.reshape(-1, 1)
    return np.mean( np.var(y_model, axis=1, keepdims=True) )


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


class OLS:
    def __init__(self) -> None:
        pass    

    def __str__(self) -> str:
        return 'OLS'
        
    def fit(self, x: np.ndarray, y: np.ndarray):
        y = np.ravel(y)
        self.coef_ = (np.linalg.inv(x.T @ x) @ x.T ) @ y

        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return (x @ self.coef_).reshape(-1,1)
    

class Ridge:
    def __init__(self, alpha: float = 0.1) -> None:
        self.param = {'alpha': alpha}

    def __str__(self) -> str:
        return 'Ridge'
    
    def set_params(self, alpha):
        self.param['alpha'] = alpha
        
    def fit(self, x: np.ndarray, y: np.ndarray):
        n = x.shape[1]
        I = np.eye(n, n)
        lmbda = self.param['alpha']
        y = np.ravel(y)
        self.coef_ = np.linalg.inv(x.T @ x + lmbda*I) @ x.T @ y

        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return (x @ self.coef_).reshape(-1,1)


class LinearRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, model) -> None:
        self.x = x
        self.y = y
        self.z = z

        self.nx, self.ny = z.shape

        self.name = model
        if model == 'OLS':
            self.model = linear_model.LinearRegression(fit_intercept=False)
        elif model == 'Ridge':
            self.model = linear_model.Ridge(fit_intercept=False)
        elif model == 'Lasso':
            self.model = linear_model.Lasso(fit_intercept=False)
        elif isinstance(model, Callable):
            self.model = model()
            self.name = model.__name__
        else:
            self.model = model

        self.params: dict[str, any] = {
            'p': [],
            'lmbda': []
        }

    def SetModel(self, model):
        self.name = model
        if model == 'OLS':
            self.model = linear_model.LinearRegression(fit_intercept=False)
        elif model == 'Ridge':
            self.model = linear_model.Ridge(fit_intercept=False)
        elif model == 'Lasso':
            self.model = linear_model.Lasso(fit_intercept=False)
        elif isinstance(model, Callable):
            self.model = model()
            self.name = model.__name__
        else:
            self.model = model
    
    def DesignMatrix(self, p: int, intercept: bool = True) -> np.ndarray:
        if self.y.any() != None:
            x, y = np.ravel(self.x), np.ravel(self.y)
            self.X = np.ones((len(x), 1))
            for k in range(1, p + 1):
                j = k
                i = 0
                while i < k + 1:
                    self.X = np.column_stack((self.X, x**i * y**j))
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
            
    def Plot(self, plot: str = 'MSE', ls: str = '-') -> None:
        fig ,axs = plt.subplots(1, 2, sharex=True, sharey=True)
        
        # axs[0].imshow((self.X @ self.beta).reshape(self.nx, self.ny), cmap=cm.coolwarm, origin='lower')
        axs[0].imshow(self.model.predict(self.X).reshape(self.nx, self.ny), cmap=cm.coolwarm, origin='lower')
        axs[1].imshow(self.z, cmap=cm.coolwarm, origin='lower')
        axs[0].set_xlabel('X')
        axs[1].set_xlabel('X')
        axs[0].set_ylabel('Y')

        axs[0].set_title('Fitted data')
        axs[1].set_title('Real data')
        
        fig.savefig(f"Plots/{self.name}.pdf")
        
    def PlotGraph(self, plot: str = 'MSE', ls: str = '-', train: bool = False) -> None:
        plt.grid()
        plt.xlabel('lmbda')
        plt.ylabel(plot)
        plt.tight_layout(rect=[0.05, 0, 0.85, 1])

        if plot == 'MSE':
            y = self.results['MSE_test']
            y2 = self.results['MSE_train']
        else:
            y = self.results[plot]

        if isinstance(self.key, tuple):
            k1, k2 = self.key
            if k1 == 'lmbda':
                k1, k2 = k2, k1
            p1, p2 = self.params[k1], self.params[k2]
            plt.xscale('log')
            for i in range(len(p1)):
                plt.plot(p2, y[i,:], label=f"p = {i}", ls=ls)
                
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(f"Plots/{self.name}_lmbda+p_{plot}.pdf", bbox_inches='tight')
            
            
        else:
            if self.key == 'lmbda':
                plt.xscale('log')
            if train:
                plt.plot(self.params[self.key], y2, label=f'Train', ls=ls)
            plt.plot(self.params[self.key], y, label=f'Test', ls=ls)
            
            plt.ylabel(plot)
            plt.xlabel(self.key)
            plt.legend()
            
            plt.savefig(f"Plots/{self.name}_{self.key}_{plot}.pdf")        
        
    def PlotHeatMap(self, plot: str = 'MSE', ls: str = '-') -> None:
        fig, ax = plt.subplots(figsize=(14,14))
        lambdas = self.params['lmbda']

        sns.heatmap(
            np.swapaxes(self.MSE_test, 0, 1),
            cmap='viridis',
            square=True,
            robust=True,
            annot=True,
            yticklabels=[f'{np.log10(lmbda):.2f}' for lmbda in lambdas],
            cbar_kws={'label': 'Mean Squared Error'},
        )

        ax.set_xlabel('p')
        ax.set_ylabel(r'log$_{10}$($\beta$)')

        # cont = plt.contourf(self.params['p'], self.params['lmbda'], np.swapaxes(self.MSE_test, 0, 1), norm='log')
        # cont = plt.imshow(np.log(z), cmap=cm.coolwarm, origin='lower', extent=(x[0], x[-1], y[0], y[-1]))
        # plt.colorbar(cont, aspect=5)
        
        plt.savefig('Plots/'+self.name+"_HM.pdf")

    def PlotBeta(self, plot: str = 'MSE', ls: str = '-') -> None:
        fig, ax = plt.subplots(figsize=(16,4))
        mask = np.where(self.beta == np.inf, 1, 0)
        sns.heatmap(
            self.beta, 
            mask=mask, 
            cmap='viridis',
            square=True, 
            robust=True,
            annot=True,
            cbar_kws={'label': 'Mean Squared Error'}
        )

        ax.set_ylabel('p')
        ax.set_xlabel(r'number of $\beta$ values')

        plt.savefig(f'Plots/{self.name}_BetaHM.pdf')

    def BiasVariance(self, plot: str = 'MSE', ls: str = '-') -> None:
        mse = self.results['MSE_test']
        bias = self.results['Bias']
        variance = self.results['Variance']

        if isinstance(self.key, tuple):
            pass
        else:
            x = self.params[self.key]

            plt.plot(x, mse, label='Error')
            plt.plot(x, bias, label='Bias')
            plt.plot(x, variance, label='Variance')

        plt.legend()
        plt.grid()
        plt.savefig(f"Plots/{self.name}_BiasVariance.pdf")

    def TrainAndTest(self, *params, p: int = 5, lmbda: float = None, rng: int = None, bootstrap: int = 0, scale_data: bool = False) -> None:
        X_train, X_test, y_train, y_test = train_test_split(self.X, np.ravel(self.z), test_size=0.2, random_state=rng)

        y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)

        if scale_data:
            xscaler = StandardScaler()
            yscaler = StandardScaler()
            zshape = z.shape

            xscaler.fit(X_train)
            X_train = xscaler.transform(X_train)
            X_test = xscaler.transform(X_test)
            self.X = xscaler.transform(self.X)

            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            self.z = z.ravel().reshape(-1, 1)

            yscaler.fit(y_train)
            y_train = np.ravel(yscaler.transform(y_train)).reshape(-1, 1)
            y_test = np.ravel(yscaler.transform(y_test)).reshape(-1, 1)
            self.z = np.ravel(yscaler.transform(self.z)).reshape(zshape)

        if lmbda != None:
            self.model.set_params(alpha=lmbda)

        if len(params) == 1:
            self.key = params[0]
            param = self.params[params[0]]
            self.MSE_test = np.zeros(len(param))
            self.MSE_Train = np.zeros(len(param))
            self.R2 = np.zeros(len(param))
            self.Bias = np.zeros(len(param))
            self.Variance = np.zeros(len(param))

            pn = len(self.params['p'])
            self.beta = np.full((pn, int(((pn + 1)*(pn))/2)), np.inf)
            for i in range(len(param)):
                X_train_, X_test_ = X_train, X_test
                if self.key == 'p':
                    n = int(((i + 2)*(i + 1))/2)
                    X_train_, X_test_ = X_train[:,:n], X_test[:,:n]
                elif self.key == 'lmbda':
                    self.model.set_params(alpha=param[i])

                self.model.fit(X_train_, y_train)

                y_fit = self.model.predict(X_train_)
                y_pred = self.model.predict(X_test_) if bootstrap == 0 else np.empty((y_test.shape[0], bootstrap))
                beta = self.model.coef_ if bootstrap == 0 else np.zeros((n, bootstrap))
                for k in range(bootstrap):
                    x, y = resample(X_train_, y_train)
                    
                    y_pred[:, k] = self.model.fit(x, y).predict(X_test_).ravel()
                    beta[:,k]= self.model.coef_
                    
                if bootstrap != 0 :
                    beta = np.mean(beta, axis=1)
                    # y_test = np.swapaxes(y_test.reshape(1, -1), 0, 1)

                self.MSE_Train[i] = MSE(y_train, y_fit)
                self.MSE_test[i] = MSE(y_test, y_pred)
                self.R2[i] = R2(y_test, y_pred)
                
                self.Bias[i] = Bias(y_test, y_pred)
                self.Variance[i] = Variance(y_pred)

                self.beta[i, :n] = beta

                # print(f'{'MSE:' : <4} {mean_squared_error(y_test, y_pred):g}')
                # print(f'{'R2:' : <4} {r2_score(y_test, y_pred):g}')

        elif len(params) == 2:
            self.key = params[0], params[1]
            p = self.params[self.key[0]], self.params[self.key[1]]

            self.MSE_test = np.zeros( ( (
                len(p[0]), 
                len(p[1])
                ) ) )
            self.MSE_Train = np.zeros( ( (
                len(p[0]), 
                len(p[1])
                ) ) )
            self.R2 = np.zeros( ( (
                len(p[0]), 
                len(p[1])
                ) ) )
            self.Bias = np.zeros( ( (
                len(p[0]), 
                len(p[1])
                ) ) )
            self.Variance = np.zeros( ( (
                len(p[0]), 
                len(p[1])
                ) ) )
            
            for i in range(len(p[0])):
                X_train_, X_test_ = X_train, X_test
                if self.key[0] == 'p':
                    n = int(((i + 2)*(i + 1))/2)
                    X_train_, X_test_ = X_train[:,:n], X_test[:,:n]
                elif self.key[0] == 'lmbda':
                    self.model.set_params(alpha=p[0][i])

                for j in range(len(p[1])):
                    if self.key[1] == 'p':
                        n = int(((j + 2)*(j + 1))/2)
                        X_train_, X_test_ = X_train[:,:n], X_test[:,:n]
                    elif self.key[1] == 'lmbda':
                        self.model.set_params(alpha=p[1][j])

                    self.model.fit(X_train_, y_train)

                    y_fit = self.model.predict(X_train_)
                    y_pred = self.model.predict(X_test_) if bootstrap == 0 else np.empty((y_test.shape[0], bootstrap))
                    beta = self.model.coef_ if bootstrap == 0 else np.zeros((n, bootstrap))
                    for k in range(bootstrap):
                        x, y = resample(X_train_, y_train)
                        
                        y_pred[:, k] = self.model.fit(x, y).predict(X_test_).ravel()
            
                    if bootstrap != 0 :
                        beta = np.mean(beta, axis=1)
                        # y_test = np.swapaxes(y_test.reshape(1, -1), 0, 1)

                    self.MSE_Train[i, j] = MSE(y_train, y_fit)
                    self.MSE_test[i, j] = MSE(y_test, y_pred)
                    self.R2[i, j] = R2(y_test, y_pred)
                    self.Bias[i, j] = Bias(y_test, y_pred)
                    self.Variance[i, j] = Variance(y_pred)

                    # print(f'{'MSE:' : <4} {mean_squared_error(y_test, y_pred):g}')
                    # print(f'{'R2:' : <4} {r2_score(y_test, y_pred):g}')

        else:
            self.model.fit(X_train, y_train)
            n = int(((p + 2)*(p + 1))/2)

            y_fit = self.model.predict(X_train)
            y_pred = self.model.predict(X_test) if bootstrap == 0 else np.empty((y_test.shape[0], bootstrap))
            self.beta = self.model.coef_ if bootstrap == 0 else np.zeros((n, bootstrap))
            for k in range(bootstrap):
                x, y = resample(X_train, y_train)
                
                y_pred[:, k] = self.model.fit(x, y).predict(X_test).ravel()
                self.beta[:,k] = self.model.coef_
            
            if bootstrap != 0 :
                self.beta = np.mean(self.beta, axis=1)
                # y_test = np.swapaxes(y_test.reshape(1, -1), 0, 1)

            print(self.beta.shape, self.beta)

            # y_test = y_test.reshape(-1,1)
            self.MSE_Train = MSE(y_train, y_fit)
            self.MSE_test = MSE(y_test, y_pred)
            self.R2 = R2(y_test, y_pred)
            self.Bias = Bias(y_test, y_pred)
            self.Variance = Variance(y_pred)

            print(f'{"MSE:" : <10} {self.MSE_test:g}')
            print(f'{"R2:" : <10} {self.R2:g}')
            print(f'{"Bias:" : <10} {self.Bias:g}')
            print(f'{"Variance:" : <10} {self.Variance:g}')

        # print(self.X.shape, self.beta.shape)
        # print(self.X)
        # print(self.beta)
        # print((self.X @ self.beta).reshape(100, 100))

        self.results = {
            'MSE_test': self.MSE_test,
            'MSE_train': self.MSE_Train,
            'R2': self.R2,
            'Bias': self.Bias,
            'Variance': self.Variance,
        }

    def CrossValidation(self, *params, p_order: int = 5, lmbda: float = None, k: int = 10, n_jobs: int = 1) -> None:
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

        cv = GridSearchCV(Estimator(lin_reg=self.name, p=p_order, lmbda=lmbda),
                      cv=k,
                      param_grid=param_grid,
                      scoring='neg_mean_squared_error',
                      n_jobs=n_jobs)
    
        cv.fit(self.X, np.ravel(self.z))

        """print(cv.best_index_)
        df = pd.DataFrame(cv.cv_results_)
        MSE_std = df['rank_test_score' == cv.best_index_]
        print(pd.DataFrame(MSE_std))"""
        
        print(f"{'Parameter' : <12} Value")
        print('---------------------------')
        for param, value in cv.best_params_.items():
            print(f"{'Best '+param : <12} {value:g}")
        print(f"{'Best MSE' : <12} {-cv.best_score_:.5f}")
        # print(f"{'CV score': <12} {-np.mean(cross_val_score(cv.best_estimator_, self.X, self.z, scoring='neg_mean_squared_error', cv=k)):.5f}")

        n = len(params)
        results = -cv.cv_results_['mean_test_score']
        if n == 1:
            self.key = params[0]
            self.MSE_test = results
        elif n > 1:
            results = results.reshape(len(self.params['p']), len(self.params['lmbda']), order='F')
            self.key = params
            self.MSE_test = results

        self.results = {
            'MSE': self.MSE_test
        }
        self.name += '_CV'
        

if __name__ == "__main__":
    start = time.time()
    n = 30
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    # x = np.sort(np.random.uniform(0.1, 1, n))
    # y = np.sort(np.random.uniform(0.1, 1, n))

    x, y = np.meshgrid(x, y)
        
    z = FrankeFunction(x, y, 0.05)
    
    p = 17
    lmbda = 1e-2
    
    lambdas = np.logspace(-8, 1, 20)
    ps = np.arange(p+1)
    
    k = 10
    # ridge = LinearRegression(x, y, z, 'Ridge')
    # ridge.DesignMatrix(p)
    # ridge.SetParams(p=ps, lmbda=lambdas)
    # ridge.TrainAndTest('p', 'lmbda', bootstrap=10, p=p, lmbda=lmbda)
    # ridge.CrossValidation('p', 'lmbda', n_jobs=8)

    # ridge.Plot()
    # ridge.PlotGraph('R2')
    # plt.figure()
    # ridge.PlotGraph('MSE')
    # plt.figure()
    # ridge.PlotBeta()
    # plt.figure()
    # ridge.PlotHeatMap()
    # ridge.BiasVariance()

    ols = LinearRegression(x, y, z, 'OLS')
    ols.DesignMatrix(p)
    ols.SetParams(p=ps, lmbda = lambdas)
 
    ols.TrainAndTest('p', p=p, bootstrap=40, lmbda=None, scale_data=True)
    ols.Plot()
    plt.figure()
    ols.PlotGraph('MSE', train=True)
    # plt.figure()
    # ols.PlotGraph('R2')

    # ols.SetModel('Ridge')
    # ols.TrainAndTest('p', p=p, bootstrap=0, lmbda=lmbda)
    # ols.PlotGraph('MSE', ls='--', label='sklearn', train=True)

    # plt.figure()
    # ols.PlotBeta()
    plt.figure()
    ols.BiasVariance()
    
    # ols.CrossValidation('p', name='OLS_CV', n_jobs=8)
    # ols.PlotGraph('MSE')


    end = time.time()
    print(f'Time: {end - start:.2f}s')
    
    # plt.show()