from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

SEARCH_SPACE = {
    "hidden_layer_sizes": [(40, 40), (50, 50), (30, 30)],
    "activation": ["tanh", "relu", "logistic"],
    "solver": ["lbfgs", "sgd", "adam"],
    "alpha": [1e-3, 1e-2, 1e-1],
    "learning_rate": ["constant", "adaptive"],
    "learning_rate_init": [1e-2, 1e-3],
}

def grid_search_mlpr(x_train, y_train, cv=5, n_jobs=-1, max_iter=400):
    base = MLPRegressor(max_iter=max_iter)
    gs = GridSearchCV(estimator=base, param_grid=SEARCH_SPACE, n_jobs=n_jobs, cv=cv)
    gs.fit(x_train, y_train)
    return gs