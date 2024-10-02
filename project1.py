import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from imageio import imread
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed


class LinearRegression:
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, z) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    def DesignMatrix(self, p: int, intercept: bool = True):
        if self.y.any() != None:
            self.X = np.ones((len(self.x, p + 1)))
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