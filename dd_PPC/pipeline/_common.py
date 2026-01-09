__all__ = ['fit_and_predictions_model', 'pred_model', 'fit_and_test_model']

import random

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .. import file, preprocess, model, data, calc


_GLOBAL_LAMBDA = 0.09


def fit_and_test_model(model_names: list[str], model_params: dict | None = None, boxcox_lambda: float | None = None):
    """Fits and tests the selected_model; evaluates competition score"""

    if boxcox_lambda is None:
        boxcox_lambda = _GLOBAL_LAMBDA

    def fit_data(train_x_, train_cons_y_, train_rate_y_):

        _x_train, sc, _ = _preprocess_data(train_x_)
        _y_train = _get_modified_target(train_cons_y_, boxcox_lambda)

        models, pred_vals = [], []
        for _model in model_names:
            _one_models, _one_pred_vals = _modeling_with_some_seeds(_model, model_params, _x_train, _y_train, boxcox_lambda)
            models.extend(_one_models)
            pred_vals.extend(_one_pred_vals)

        consumption = train_cons_y_.copy()
        consumption['cons_pred'] = np.mean(pred_vals, axis=0)

        pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        ir = model.fit_isotonic_regression(pred_rate_y, train_rate_y_)

        _transformed_rate_y = model.transform_isotonic_regression(pred_rate_y, ir)

        print('train comp score:', calc.weighted_average_of_consumption_and_poverty_rate(consumption, train_rate_y_, _transformed_rate_y))

        return models, pred_vals, sc, ir


    def pred_data(test_x_, test_cons_y_, sc: StandardScaler, models: list, ir: IsotonicRegression):

        x_test, *_ = _preprocess_data(test_x_, sc)
        pred_cons_ys = _fitting_with_some_models(models, x_test, boxcox_lambda)

        pred_cons_y = np.mean(pred_cons_ys, axis=0)

        y_test = test_cons_y_.loc[:, 'cons_ppp17']

        consumption = test_cons_y_.copy()
        consumption['cons_pred'] = pred_cons_y

        pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        pred_rate_y = model.transform_isotonic_regression(pred_rate_y, ir)

        return x_test, y_test, consumption, pred_cons_y, pred_rate_y

    def show_metrics(pred_cons_y, y_test, pred_rate_y, consumption, models: list, x_test, test_rate_y_):
        print(f'RMSE: {np.sqrt(np.mean((pred_cons_y - y_test) ** 2))}')
        print(f'MAE: {np.mean(np.abs(pred_cons_y - y_test))}')
        print(f'R2: {np.mean([_lb.score(x_test, y_test) for _lb in models])}')
        print(
            f'CompetitionScore: {calc.weighted_average_of_consumption_and_poverty_rate(consumption, pred_rate_y, test_rate_y_)}')

    _datas = file.get_datas()

    train_x, train_cons_y, train_rate_y, test_x, test_cons_y, test_rate_y = data.split_datas(_datas['train'],
                                                                                  _datas['target_consumption'],
                                                                                  _datas['target_rate'])

    _models, _pred_vals, _sc, _ir = fit_data(train_x, train_cons_y, train_rate_y)

    _x_test, _y_test, _consumption, _pred_cons_y, _pred_rate_y = pred_data(test_x, test_cons_y, _sc, _models, _ir)

    show_metrics(_pred_cons_y, _y_test, _pred_rate_y, _consumption, _models, _x_test, test_rate_y)


def fit_and_predictions_model(model_name, folder_prefix: str | None = None):
    """Fits the model; predicts consumption; saves the submission format"""

    _datas = file.get_datas()

    # learning
    _x_train, _sc, _cat_cols = _preprocess_data(_datas['train'])
    _y_train = _get_modified_target(_datas['target_consumption'])

    if model_name == 'lightgbm':
        _model, _cons_pred = getattr(model, f'fit_{model_name}')(_x_train, _y_train, categorical_cols=_cat_cols)
    else:
        _model, _cons_pred = getattr(model, f'fit_{model_name}')(_x_train, _y_train)

    _cons_pred = calc.inverse_boxcox_transform(_cons_pred, _GLOBAL_LAMBDA)

    _consumption = _datas['target_consumption']
    _consumption['cons_pred'] = _cons_pred
    pred_rate_y = calc.poverty_rates_from_consumption(_consumption, 'cons_pred')
    ir = model.fit_isotonic_regression(pred_rate_y, _datas['target_rate'])

    # prediction
    _predicted_coxbox = pred_model(_model, _sc)
    _predicted = calc.inverse_boxcox_transform(_predicted_coxbox, _GLOBAL_LAMBDA)

    _consumption, _ = file.get_submission_formats('../results')
    _consumption['cons_pred'] = _predicted
    pred_rate_y = calc.poverty_rates_from_consumption(_consumption, 'cons_pred')
    pred_rate_y = model.transform_isotonic_regression(pred_rate_y, ir)

    file.save_to_submission_format(_predicted, pred_rate=pred_rate_y, folder_prefix=folder_prefix)


def pred_model(fit_model, sc: StandardScaler) -> np.ndarray:
    _datas = file.get_datas()

    _datas_std, _ = preprocess.standardized_with_numbers_dataframe(_datas['test'], sc)
    _datas_category = preprocess.encoding_category_dataframe(_datas['test'])

    return fit_model.predict(pd.concat([_datas_std, _datas_category], axis=1))


def _preprocess_data(datas: pd.DataFrame, sc: StandardScaler | None = None) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    _datas_std, sc = preprocess.standardized_with_numbers_dataframe(datas, sc)
    _datas_aggregates = preprocess.create_survey_aggregates(datas)
    _datas_category = preprocess.encoding_category_dataframe(datas)

    category_cols = _datas_category.columns

    return pd.concat([_datas_std, _datas_category, _datas_aggregates], axis=1), sc, list(category_cols)


def _get_modified_target(targets: pd.DataFrame, boxcox_lambda: float | None = None) -> pd.DataFrame:

    if boxcox_lambda is None:
        boxcox_lambda = _GLOBAL_LAMBDA

    return calc.apply_boxcox_transform(targets.loc[:, 'cons_ppp17'], boxcox_lambda)[0]


def _modeling_with_some_seeds(model_name: str, model_params: dict | None, x_train, y_train, boxcox_lambda: float) -> tuple[list, list[np.ndarray]]:
    random.seed(0)
    _seeds_length = 2

    seed_list = [123] + random.sample(range(1, 1000), _seeds_length)
    # seed_list = [123]
    model_with_preds = [
        getattr(model, f'fit_{model_name}')(x_train, y_train, seed=_seed, params=model_params)
        for _seed in tqdm(seed_list, desc=f'{model_name}: modeling with some seeds')
    ]
    models = [_model for _model, _ in model_with_preds]
    preds = [calc.inverse_boxcox_transform(_preds_boxcox, boxcox_lambda) for _, _preds_boxcox in model_with_preds]

    return models, preds


def _fitting_with_some_models(models, x_test, boxcox_lambda: float) -> list[np.ndarray]:
    _preds_boxcox = [_model.predict(x_test) for _model in models]

    return [calc.inverse_boxcox_transform(_pred_bc, boxcox_lambda) for _pred_bc in _preds_boxcox]

