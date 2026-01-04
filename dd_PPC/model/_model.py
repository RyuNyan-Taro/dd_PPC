__all__ = ['fit_random_forest', 'fit_lightgbm']

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

def fit_random_forest(x_train_std, y_train, show_fit_process: bool = True, show_pred_plot: bool = False, seed: int = 42) -> tuple[RandomForestRegressor, np.ndarray]:
    _verbose = 2 if show_fit_process else 0

    RF = RandomForestRegressor(verbose=_verbose, random_state=seed, n_estimators=150)
    RF.fit(x_train_std, y_train)

    pred_RF = RF.predict(x_train_std)

    if show_pred_plot:
        plt.scatter(y_train, pred_RF - y_train)
        plt.xlabel('y_train')
        plt.ylabel('difference from y_train')
        plt.show()

    return RF, pred_RF


def fit_lightgbm(x_train, y_train, seed: int = 42, show_pred_plot: bool = False) -> tuple[lgb.LGBMRegressor, np.ndarray]:
    model = lgb.LGBMRegressor(random_state=seed, verbose=1, n_estimators=3000)
    pred_y = model.fit(x_train, y_train)

    pred_lgb = pred_y.predict(x_train)

    if show_pred_plot:
        plt.scatter(y_train, pred_lgb - y_train)
        plt.xlabel('y_train')
        plt.ylabel('difference from y_train')
        plt.show()

    return pred_y, pred_lgb