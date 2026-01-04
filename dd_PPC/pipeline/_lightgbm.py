__all__ = ['apply_lightgbm', 'fit_and_predictions_lightgbm', 'pred_lightgbm', 'fit_and_test_lightgbm']

import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .. import file, preprocess, model, data, calc


def apply_lightgbm(show_pred_plot: bool = False) -> tuple[lgb.LGBMRegressor, np.ndarray, StandardScaler]:
    _datas = file.get_datas()

    _datas_std, sc = preprocess.standardized_with_numbers_dataframe(_datas['train'])
    _datas_category = preprocess.encoding_category_dataframe(_datas['train'])

    _x_train = pd.concat([_datas_std, _datas_category], axis=1)
    _y_train = np.log1p(_datas['target_consumption'].loc[:, 'cons_ppp17'])

    LB, pred_LB_log = model.fit_lightgbm(_x_train, _y_train, show_pred_plot=show_pred_plot)

    pred_LB = np.expm1(pred_LB_log)

    return LB, pred_LB, sc


def fit_and_test_lightgbm():
    """Fits and tests LightGBM model; evaluates competition score"""

    def fit_data(train_x_, train_cons_y_):
        _datas_std, sc = preprocess.standardized_with_numbers_dataframe(train_x_)
        _datas_category = preprocess.encoding_category_dataframe(train_x_)

        _cat_cols = _datas_category.columns.tolist()

        _x_train = pd.concat([_datas_std, _datas_category], axis=1)
        _y_train = np.log1p(train_cons_y_.loc[:, 'cons_ppp17'])

        LB, pred_LB_log = model.fit_lightgbm(_x_train, _y_train, categorical_cols=_cat_cols)

        pred_LB = np.expm1(pred_LB_log)

        return LB, pred_LB, sc


    def pred_data(test_x_, test_cons_y_, sc, lb):
        _datas_std, sc = preprocess.standardized_with_numbers_dataframe(test_x_, sc)
        _datas_category = preprocess.encoding_category_dataframe(test_x_)

        x_test = pd.concat([_datas_std, _datas_category], axis=1)

        _pred_cons_y_log = lb.predict(x_test)

        pred_cons_y = np.expm1(_pred_cons_y_log)

        y_test = test_cons_y_.loc[:, 'cons_ppp17']

        consumption = test_cons_y_.copy()
        consumption['cons_pred'] = pred_cons_y

        pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        return x_test, y_test, consumption, pred_cons_y, pred_rate_y

    def show_metrics(pred_cons_y, y_test, pred_rate_y, consumption, lb, x_test, test_rate_y_):
        print(f'RMSE: {np.sqrt(np.mean((pred_cons_y - y_test) ** 2))}')
        print(f'MAE: {np.mean(np.abs(pred_cons_y - y_test))}')
        print(f'R2: {lb.score(x_test, y_test)}')
        print(
            f'CompetitionScore: {calc.weighted_average_of_consumption_and_poverty_rate(consumption, pred_rate_y, test_rate_y_)}')

    _datas = file.get_datas()

    train_x, train_cons_y, _, test_x, test_cons_y, test_rate_y = data.split_datas(_datas['train'],
                                                                                  _datas['target_consumption'],
                                                                                  _datas['target_rate'])

    _LB, _pred_LB, _sc = fit_data(train_x, train_cons_y)

    _x_test, _y_test, _consumption, _pred_cons_y, _pred_rate_y = pred_data(test_x, test_cons_y, _sc, _LB)

    show_metrics(_pred_cons_y, _y_test, _pred_rate_y, _consumption, _LB, _x_test, test_rate_y)




def fit_and_predictions_lightgbm(folder_prefix: str | None = None):
    """Fits lightgbm model; predicts consumption; saves the submission format"""

    _datas = file.get_datas()

    _train = _datas['train']
    _target = _datas['target_consumption']

    _datas_std, _sc = preprocess.standardized_with_numbers_dataframe(_datas['train'])
    _datas_category = preprocess.encoding_category_dataframe(_datas['train'])

    _x_train = pd.concat([_datas_std, _datas_category], axis=1)
    _y_train = np.log1p(_datas['target_consumption'].loc[:, 'cons_ppp17'])

    _cat_cols = _datas_category.columns

    _LB, pred_LB_log = model.fit_lightgbm(_x_train, _y_train, categorical_cols=_cat_cols)

    _predicted = pred_lightgbm(_LB, _sc)

    file.save_to_submission_format(_predicted, folder_prefix)


def pred_lightgbm(fit_model: lgb.LGBMRegressor, sc: StandardScaler) -> np.ndarray:
    _datas = file.get_datas()

    _datas_std, _ = preprocess.standardized_with_numbers_dataframe(_datas['test'], sc)
    _datas_category = preprocess.encoding_category_dataframe(_datas['test'])

    pred_log = fit_model.predict(pd.concat([_datas_std, _datas_category], axis=1))

    return np.expm1(pred_log)