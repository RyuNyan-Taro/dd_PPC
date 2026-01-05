__all__ = ['apply_lightgbm', 'fit_and_predictions_lightgbm', 'pred_lightgbm', 'fit_and_test_lightgbm']

import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .. import file, preprocess, model, data, calc


_GLOBAL_LAMBDA = 0.09

def apply_lightgbm(show_pred_plot: bool = False, survey_ids: list[int] | None = None) -> tuple[lgb.LGBMRegressor, np.ndarray, StandardScaler]:
    _datas = file.get_datas()

    if survey_ids is None:
        _x = _datas['train']
        _y = _datas['target_consumption'].loc[:, 'cons_ppp17']
    else:
        _x = _datas['train'].loc[_datas['train'].survey_id.isin(survey_ids), :]
        _y = _datas['target_consumption'].loc[_datas['target_consumption'].survey_id.isin(survey_ids), 'cons_ppp17']

    _datas_std, sc = preprocess.standardized_with_numbers_dataframe(_x)
    _datas_category = preprocess.encoding_category_dataframe(_x)

    _x_train = pd.concat([_datas_std, _datas_category], axis=1)
    _y_train, _ = calc.apply_boxcox_transform(_y, _GLOBAL_LAMBDA)

    LB, pred_LB_coxbox = model.fit_lightgbm(_x_train, _y_train, show_pred_plot=show_pred_plot)

    pred_LB = calc.inverse_boxcox_transform(pred_LB_coxbox, _GLOBAL_LAMBDA)

    return LB, pred_LB, sc


def fit_and_test_lightgbm(boxcox_lambda: float | None = None):
    """Fits and tests LightGBM model; evaluates competition score"""

    if boxcox_lambda is None:
        boxcox_lambda = _GLOBAL_LAMBDA

    def fit_data(train_x_, train_cons_y_):
        _datas_std, sc = preprocess.standardized_with_numbers_dataframe(train_x_)
        _datas_category = preprocess.encoding_category_dataframe(train_x_)

        _x_train = pd.concat([_datas_std, _datas_category], axis=1)
        _y_train, _ = calc.apply_boxcox_transform(train_cons_y_.loc[:, 'cons_ppp17'], boxcox_lambda)

        LB, pred_LB_log = model.fit_lightgbm(_x_train, _y_train)

        pred_LB = calc.inverse_boxcox_transform(pred_LB_log, boxcox_lambda)

        return LB, pred_LB, sc


    def pred_data(test_x_, test_cons_y_, sc, lb):
        _datas_std, sc = preprocess.standardized_with_numbers_dataframe(test_x_, sc)
        _datas_category = preprocess.encoding_category_dataframe(test_x_)

        x_test = pd.concat([_datas_std, _datas_category], axis=1)
        _pred_cons_y_log = lb.predict(x_test)

        pred_cons_y = calc.inverse_boxcox_transform(_pred_cons_y_log, boxcox_lambda)

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
    _y_train, _ = calc.apply_boxcox_transform(_datas['target_consumption'].loc[:, 'cons_ppp17'], _GLOBAL_LAMBDA)

    _cat_cols = _datas_category.columns

    _LB, _ = model.fit_lightgbm(_x_train, _y_train, categorical_cols=_cat_cols)

    _predicted_coxbox = pred_lightgbm(_LB, _sc)

    _predicted = calc.inverse_boxcox_transform(_predicted_coxbox, _GLOBAL_LAMBDA)

    file.save_to_submission_format(_predicted, folder_prefix)


def pred_lightgbm(fit_model: lgb.LGBMRegressor, sc: StandardScaler) -> np.ndarray:
    _datas = file.get_datas()

    _datas_std, _ = preprocess.standardized_with_numbers_dataframe(_datas['test'], sc)
    _datas_category = preprocess.encoding_category_dataframe(_datas['test'])

    return fit_model.predict(pd.concat([_datas_std, _datas_category], axis=1))


def _preprocess_data(datas: pd.DataFrame, sc: StandardScaler | None) -> tuple[pd.DataFrame, StandardScaler]:
    _datas_std, sc = preprocess.standardized_with_numbers_dataframe(datas['test'], sc)
    _datas_category = preprocess.encoding_category_dataframe(datas['test'])

    return pd.concat([_datas_std, _datas_category], axis=1), sc


def _get_modified_target(targets: pd.DataFrame, boxcox_lambda: float | None = None) -> pd.DataFrame:

    if boxcox_lambda is None:
        boxcox_lambda = _GLOBAL_LAMBDA

    return calc.apply_boxcox_transform(targets['target_consumption'].loc[:, 'cons_ppp17'], boxcox_lambda)[0]
