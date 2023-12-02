from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe
import seaborn as sns
import numpy as np
import pandas as pd


def print_classif_report(model_name, pred, y):
    print(f'{model_name}: ')
    print("- roc_auc_score:", round(roc_auc_score(y.values.ravel(), pred), 4))
    print("- accuracy_score:", round(accuracy_score(y.values.ravel(), pred), 4))
    print("- f1_score:", round(f1_score(y.values.ravel(), pred), 4))
    print()


def model_fit_predict(space):
    eval_data = space['eval_data']
    model_type = space['model']
    params = space['params']

    model = model_type(**params)

    X_train = eval_data['X_train'] if type(eval_data['X_train']) == np.ndarray else eval_data['X_train'].values
    y_train = eval_data['y_train'] if type(eval_data['y_train']) == np.ndarray else eval_data['y_train'].values
    X_test = eval_data['X_test'] if type(eval_data['X_test']) == np.ndarray else eval_data['X_test'].values

    model.fit(X_train, y_train.ravel())
    pred = model.predict(X_test)
    return model, pred


def objective(space):
    metric = space['metric']
    eval_data = space['eval_data']
    _, pred = model_fit_predict(space)

    curr_metric = metric(eval_data['y_test'].values.ravel(), pred)

    return -curr_metric


def hp_optimize(trial, space, max_evals):
    # trial, space, max_evals = param
    best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trial, max_evals=max_evals,
                rstate=np.random.default_rng(42))
    return best



