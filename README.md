# fys-stk4155_project1


Runs for the project

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

     python3 main_Franke.py model=Ridge plots=graph,beta p=20 lmbda=1e-4 data=terrain1 params=p scale=1
    cv:
     python3 main_Franke.py plots=heatmap,plot model=Ridge cv=1 njobs=8 p=20 params=p data=terrain1 scale=1