__all__ = ['fit_random_forest']

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def fit_random_forest(x_train_std, y_train, show_fit_process: bool = True, show_pred_plot: bool = False):
    _verbose = 2 if show_fit_process else 0

    RF = RandomForestRegressor(verbose=_verbose)
    RF.fit(x_train_std, y_train)

    pred_RF = RF.predict(x_train_std)

    if show_pred_plot:
        plt.scatter(y_train, pred_RF - y_train)
        plt.xlabel('y_train')
        plt.ylabel('difference from y_train')
        plt.show()

    return RF, pred_RF