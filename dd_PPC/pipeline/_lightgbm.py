__all__ = ['apply_lightgbm', 'fit_and_predictions_lightgbm', 'pred_lightgbm', 'fit_and_test_lightgbm']

import random

import numpy as np
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.isotonic import IsotonicRegression
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

    _x_train, sc, _ = _preprocess_data(_x)
    _y_train = _get_modified_target(_y)

    LB, pred_LB_coxbox = model.fit_lightgbm(_x_train, _y_train, show_pred_plot=show_pred_plot)

    pred_LB = calc.inverse_boxcox_transform(pred_LB_coxbox, _GLOBAL_LAMBDA)

    return LB, pred_LB, sc


def fit_and_test_lightgbm(boxcox_lambda: float | None = None):
    """Fits and tests LightGBM model; evaluates competition score"""

    if boxcox_lambda is None:
        boxcox_lambda = _GLOBAL_LAMBDA

    def fit_data(train_x_, train_cons_y_, train_rate_y_):

        _x_train, sc, _ = _preprocess_data(train_x_)
        _y_train = _get_modified_target(train_cons_y_, boxcox_lambda)

        # LB, pred_LB_log = model.fit_lightgbm(_x_train, _y_train)
        #
        # pred_LB = calc.inverse_boxcox_transform(pred_LB_log, boxcox_lambda)

        models, pred_LBs = _modeling_with_some_seeds(_x_train, _y_train, boxcox_lambda)

        consumption = train_cons_y_.copy()
        consumption['cons_pred'] = np.mean(pred_LBs, axis=0)

        pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        ir = model.fit_isotonic_regression(pred_rate_y, train_rate_y_)

        _transformed_rate_y = model.transform_isotonic_regression(pred_rate_y, ir)

        print('train comp score:', calc.weighted_average_of_consumption_and_poverty_rate(consumption, train_rate_y_, _transformed_rate_y))

        # return LB, pred_LB, sc, ir

        return models, pred_LBs, sc, ir


    def pred_data(test_x_, test_cons_y_, sc: StandardScaler, lbs: list[LGBMRegressor], ir: IsotonicRegression):

        x_test, *_ = _preprocess_data(test_x_, sc)
        pred_cons_ys = _fitting_with_some_models(lbs, x_test, boxcox_lambda)

        pred_cons_y = np.mean(pred_cons_ys, axis=0)

        y_test = test_cons_y_.loc[:, 'cons_ppp17']

        consumption = test_cons_y_.copy()
        consumption['cons_pred'] = pred_cons_y

        pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        pred_rate_y = model.transform_isotonic_regression(pred_rate_y, ir)

        return x_test, y_test, consumption, pred_cons_y, pred_rate_y

    def show_metrics(pred_cons_y, y_test, pred_rate_y, consumption, lbs: list[LGBMRegressor], x_test, test_rate_y_):
        print(f'RMSE: {np.sqrt(np.mean((pred_cons_y - y_test) ** 2))}')
        print(f'MAE: {np.mean(np.abs(pred_cons_y - y_test))}')
        print(f'R2: {np.mean([_lb.score(x_test, y_test) for _lb in lbs])}')
        print(
            f'CompetitionScore: {calc.weighted_average_of_consumption_and_poverty_rate(consumption, pred_rate_y, test_rate_y_)}')

    _datas = file.get_datas()

    train_x, train_cons_y, train_rate_y, test_x, test_cons_y, test_rate_y = data.split_datas(_datas['train'],
                                                                                  _datas['target_consumption'],
                                                                                  _datas['target_rate'])

    _LBs, _pred_LBs, _sc, _ir = fit_data(train_x, train_cons_y, train_rate_y)

    _x_test, _y_test, _consumption, _pred_cons_y, _pred_rate_y = pred_data(test_x, test_cons_y, _sc, _LBs, _ir)

    show_metrics(_pred_cons_y, _y_test, _pred_rate_y, _consumption, _LBs, _x_test, test_rate_y)


def fit_and_predictions_lightgbm(folder_prefix: str | None = None):
    """Fits lightgbm model; predicts consumption; saves the submission format"""

    _datas = file.get_datas()

    # learning
    _x_train, _sc, _cat_cols = _preprocess_data(_datas['train'])
    _y_train = _get_modified_target(_datas['target_consumption'])

    _LB, _cons_pred = model.fit_lightgbm(_x_train, _y_train, categorical_cols=_cat_cols)
    _cons_pred = calc.inverse_boxcox_transform(_cons_pred, _GLOBAL_LAMBDA)

    _consumption = _datas['target_consumption']
    _consumption['cons_pred'] = _cons_pred
    pred_rate_y = calc.poverty_rates_from_consumption(_consumption, 'cons_pred')
    ir = model.fit_isotonic_regression(pred_rate_y, _datas['target_rate'])

    # prediction
    _predicted_coxbox = pred_lightgbm(_LB, _sc)
    _predicted = calc.inverse_boxcox_transform(_predicted_coxbox, _GLOBAL_LAMBDA)

    _consumption, _ = file.get_submission_formats('../results')
    _consumption['cons_pred'] = _predicted
    pred_rate_y = calc.poverty_rates_from_consumption(_consumption, 'cons_pred')
    pred_rate_y = model.transform_isotonic_regression(pred_rate_y, ir)

    file.save_to_submission_format(_predicted, pred_rate=pred_rate_y, folder_prefix=folder_prefix)


def pred_lightgbm(fit_model: lgb.LGBMRegressor, sc: StandardScaler) -> np.ndarray:
    _datas = file.get_datas()

    _datas_std, _ = preprocess.standardized_with_numbers_dataframe(_datas['test'], sc)
    _datas_category = preprocess.encoding_category_dataframe(_datas['test'])

    return fit_model.predict(pd.concat([_datas_std, _datas_category], axis=1))


def _preprocess_data(datas: pd.DataFrame, sc: StandardScaler | None = None) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    _datas_std, sc = preprocess.standardized_with_numbers_dataframe(datas, sc)
    _datas_category = preprocess.encoding_category_dataframe(datas)

    category_cols = _datas_category.columns

    return pd.concat([_datas_std, _datas_category], axis=1), sc, list(category_cols)


def _get_modified_target(targets: pd.DataFrame, boxcox_lambda: float | None = None) -> pd.DataFrame:

    if boxcox_lambda is None:
        boxcox_lambda = _GLOBAL_LAMBDA

    return calc.apply_boxcox_transform(targets.loc[:, 'cons_ppp17'], boxcox_lambda)[0]


def _modeling_with_some_seeds(x_train, y_train, boxcox_lambda: float) -> tuple[list[LGBMRegressor], list[np.ndarray]]:
    random.seed(0)

    seed_list = random.sample(range(1, 1000), 3)
    model_with_preds = [model.fit_lightgbm(x_train, y_train, seed=_seed, categorical_cols=None) for _seed in seed_list]
    models = [_model for _model, _ in model_with_preds]
    preds = [calc.inverse_boxcox_transform(_preds_boxcox, boxcox_lambda) for _, _preds_boxcox in model_with_preds]

    return models, preds


def _fitting_with_some_models(models, x_test, boxcox_lambda: float) -> list[np.ndarray]:
    _preds_boxcox = [_model.predict(x_test) for _model in models]

    return [calc.inverse_boxcox_transform(_pred_bc, boxcox_lambda) for _pred_bc in _preds_boxcox]

