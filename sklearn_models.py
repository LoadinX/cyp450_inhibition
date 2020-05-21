# sklearn models' construction 
## [future] liner regression for filtering outliers to be added

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from performance_evaluation import sp
PARAMETERS = {
    'svm': {'kernel': ['rbf','linear',], 
            'gamma': 2**np.arange(-3,-15,-2, dtype=float),
            #'gamma': 1.0,
            #'C':2**np.arange(-5,15,2, dtype=float),
            'C':np.arange(0.6,2.2,dtype=float),
            'class_weight':['balanced'], 
            #'class_weight':[{0:1-weight,1:weight for weight in range(0.5,0.8,0.1)}],
            'cache_size':[400] },
    'nn': {"learning_rate":["adaptive"],   #learning_rate:{"constant","invscaling","adaptive"}默认是constant
            "max_iter":[10000],
            "hidden_layer_sizes":[(100,),(400,),(600,),(800,),(1000,),(200,100),(200,100,50)],
            "alpha":10.0 ** -np.arange(1, 7),
            "activation":["relu"],  #"identity","tanh","relu","logistic"
            "solver":["adam"],     #"lbfgs" for small dataset
            'warm_start':[True]},
    'knn': {#"n_neighbors":range(2,15,1),
            "n_neighbors":range(2,10,1),
            "weights":['distance'],
            'p':[1,2],
            'metric':['minkowski','jaccard']},
    'rf': {"n_estimators":range(10,501,20),
            "criterion" : ["gini"], #['entropy']
            "oob_score": ["False"],
            "class_weight":["balanced_subsample"]
            #'class_weight':[{0:1-weight,1:weight for weight in range(0.5,0.8,0.1)}]
            }
}
SCORING_FNC = {'SE':'recall','SP':make_scorer(sp),'AUC':'roc_auc','ACC':'accuracy'}
kf = StratifiedKFold(n_splits = 5,shuffle=True,random_state = 100)

def build_sklearn_model(X, y, method_name):
    model_map = {"svm":SVC, "knn":KNeighborsClassifier,"nn":MLPClassifier,"rf":RandomForestClassifier}
    tuned_parameters = PARAMETERS[method_name]
    method = model_map[method_name]
    if method == SVC:
        grid = GridSearchCV(method(probability=True,random_state=100), 
                                    param_grid=tuned_parameters,
                                     scoring=SCORING_FNC, cv=kf, n_jobs=-1, refit='AUC' )
    elif method == KNeighborsClassifier:
        grid = GridSearchCV(method(), param_grid=tuned_parameters, 
                                    scoring =SCORING_FNC, cv=kf, n_jobs=-1, refit='AUC')
    else:
        grid = GridSearchCV(method(random_state=100), param_grid=tuned_parameters, 
                                    scoring=SCORING_FNC, cv=kf, n_jobs=-1, refit='AUC')
    grid.fit(X, y)
    return grid.best_estimator_ , grid.best_params_	,grid.cv_results_

def get_performance(cv_res):
    df = pd.DataFrame(cv_res)
    r = df.sort_values(by='rank_test_AUC').iloc[0]
    return r[['mean_test_AUC','mean_test_SE', 'mean_test_SP', 'mean_test_ACC']]
