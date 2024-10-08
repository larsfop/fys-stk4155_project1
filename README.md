# fys-stk4155_project1


Runs for the project

### Ordinary Least Squares:
    Fit the Franke function with OLS using sklearn
     python3 main_Franke.py plots=graph,beta
    Then with own method
     python3 main_Franke.py plots=graph,beta model=OLS_own

    Bootstrap:
     python3 main_Franke.py plots=bvt model=OLS bootstrap=100 p=20


### Ridge Regression
    Fit the Franke function with Ridge using sklearn
     python3 main_Franke.py model=Ridge p=15 params=p plots=graph lmbda=1e-4
    Using my own implementation
     python3 .\main_Franke.py model=Ridge p=15 params=p plots=graph lmbda=1e-4
    Checking for both p'th order polynomial and lambdas 
     python3 main_Franke.py model=Ridge p=15 params=p,lmbda plots=graph,heatmap

    Bootstrap:
     python3 main_Franke.py model=Ridge p=20 params=p plots=bvt lmbda=1e-4 bootstrap=100