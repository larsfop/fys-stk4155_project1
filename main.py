
from LinearRegression import *
import sys

start = time.time()

# Base configurations, change in terminal with key=value
config = {
    'model': 'OLS',
    'n': 30,
    'noise': 0,
    'p': 5,
    'lmbda': None,
    'nlambdas': 20,
    'bootstrap': 0,
    'cv': False,
    'params': 'p',
    'rng': None,
    'kfold': 10,
    'njobs': 1,
    'plots': 'graph',
    'data': 'Franke',
    'scale': False,
}

if len(sys.argv) > 1:
    args = [arg.split('=') for arg in sys.argv[1:]]
    for key, value in args:
        if key == 'n':
            config[key] = int(value)
        elif key == 'noise':
            config[key] = float(value)
        elif key == 'p':
            config[key] = int(value)
        elif key == 'lmbda':
            config[key] = float(value)
        elif key == 'nlambdas':
            config[key] = int(value)
        elif key == 'bootstrap':
            config[key] = int(value)
        elif key == 'cv':
            config[key] = bool(value)
        elif key == 'rng':
            config[key] = int(value)
        elif key == 'kfold':
            config[key] = int(value)
        elif key == 'njobs':
            config[key] = int(value)
        elif key == 'scale':
            config[key] = bool(value)
        else:
            config[key] = value


p = config['p']
model_own = {
    'OLS_own': OLS,
    'Ridge_own': Ridge,
}

params = {
    'p': np.arange(p+1),
    'lmbda': np.logspace(-8, 1, config['nlambdas'])
}

### DATA ###
if config['data'] == 'Franke':
    x = np.linspace(0, 1, config['n'])
    y = np.linspace(0, 1, config['n'])

    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y, config['noise'])
else:
    terrain = config['data']
    z = np.asarray(imread(f'SRTM_data_Norway_{terrain}.tif'), dtype=np.float32)[:100, :100]
    yn, xn = z.shape

    x = np.linspace(0, 1, xn, dtype=np.float32)
    y = np.linspace(0, 1, yn, dtype=np.float32)

    x, y = np.meshgrid(x, y)

### Setup Regression model ###
model = config['model'] if config['model'].split('_')[-1] != 'own' else model_own[config['model']]
reg = LinearRegression(x, y, z, model)


### Setup parameters and Design matrix ###
reg.DesignMatrix(config['p'])
reg.SetParams(**params)

### Fit model ###
if not config['cv']:
    reg.TrainAndTest(*config['params'].split(','), 
                     p=config['p'], lmbda=config['lmbda'],
                     bootstrap=config['bootstrap'], 
                     rng=config['rng'],
                     scale_data=config['scale'],
    )
else:
    reg.CrossValidation(*config['params'].split(','), 
                        p_order=config['p'],
                        lmbda=config['lmbda'],
                        k=config['kfold'],
                        n_jobs=config['njobs'],
                        scale_data=config['scale']
    )

### Make plots ###
plots = {
    'plot': reg.Plot,
    'graph': reg.PlotGraph,
    'beta': reg.PlotBeta,
    'bvt': reg.BiasVariance,
    'heatmap': reg.PlotHeatMap,
}

make_plots = config['plots'].split(',')
for p in make_plots:
    if p == 'graph':
        plots[p]('MSE', train=True)
        plt.figure()
        plots[p]('R2')
    else:
        plots[p]()

    plt.figure()

end = time.time()

print(f'Time: {(end-start):.2f}s')