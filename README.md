# fys-stk4155_project1

This project comes with two scripts

    1. LinearRegression.py
        This script contains all of the methods used in a single class, the class contains two methods, 
        one for fitting data with or without bootstrap and one that does a gridsearch with cross validation.
        The GridSearch can be ran with no parameters allow for a single cross validation if needed.
        The class contains a few plot functions that are all serve a very specific purpose.
        Finally the script contains functions for computing the MSE, R2, Bias and Variance,
        as well as seperate classes for OLS and Ridge implementations and a custom estimator for grid search

    2. main.py **kwargs
        This script runs LinearRegression.py and gains all of the results. The script can take in a few command line 
        agruments all of which can be found defined in the config dictionary inside the script,
        see below for examples for how to use them. The script is build into 5 steps:
            1. Load data and set options
            2. Setup the regression model
            3. Setup parameters and Design matrix
            4. Fit the model either with or without bootstrap, or with GridSearchCV
            5. Make plots

### Ordinary Least Squares:
    Fit the Franke function with OLS using sklearn
     python3 main_Franke.py plots=graph,beta
    Then with own method
     python3 main_Franke.py plots=graph,beta model=OLS_own

    Bootstrap:
     python3 main_Franke.py plots=bvt model=OLS bootstrap=100 p=20

    cv:
     python3 main_Franke.py plots=heatmap model=OLS cv=1 njobs=8 p=20 noise=0 params=p


### Ridge Regression
    Fit the Franke function with Ridge using sklearn
     python3 main_Franke.py model=Ridge p=15 params=p plots=graph lmbda=1e-4
    Using my own implementation
     python3 .\main_Franke.py model=Ridge p=15 params=p plots=graph lmbda=1e-4
    Checking for both p'th order polynomial and lambdas 
     python3 main_Franke.py model=Ridge p=15 params=p,lmbda plots=graph,heatmap

    Bootstrap:
     python3 main_Franke.py model=Ridge p=20 params=p plots=bvt lmbda=1e-4 bootstrap=100

    cv: with varying noise, 0,0.05,0.1
     python3 main_Franke.py plots=heatmap model=Ridge cv=1 njobs=8 p=20 noise=0 params=p,lmbda


### Lasso regression

    bootstrap:
     python3 main_Franke.py model=Lasso p=20 params=p plots=bvt,plot lmbda=1e-4 bootstrap=100

    cv: noise=0,0.05,0.1
        python3 main_Franke.py plots=heatmap model=Lasso cv=1 njobs=8 p=20 noise=0 params=p,lmbda


## Real Data

### OLS
    scaled vs unscaled:
     python3 main_Franke.py plots=graph,beta p=5 data=terrain1 params=p scale=1
     python3 main_Franke.py plots=graph,beta p=5 data=terrain1 params=p

    fits:
     python3 main_Franke.py plots=graph p=20 data=terrain1 params=p scale=1

    cv:
     python3 main_Franke.py plots=heatmap,plot model=OLS cv=1 njobs=8 p=20 params=p data=terrain1 scale=1


### Ridge
    fits:
     python3 main_Franke.py model=Ridge plots=graph,beta p=20 lmbda=1e-4 data=terrain1 params=p scale=1
    cv:
     python3 main_Franke.py plots=heatmap,plot model=Ridge cv=1 njobs=8 p=20 params=p data=terrain1 scale=1
