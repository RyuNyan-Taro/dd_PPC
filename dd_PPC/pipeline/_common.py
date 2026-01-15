__all__ = ['fit_and_predictions_model', 'pred_models', 'fit_and_test_model', 'preprocess_data']

import random

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .. import file, preprocess, model, data, calc


_GLOBAL_LAMBDA = 0.09


def fit_and_test_model(
        model_names: list[str],
        model_params: dict | None = None,
        boxcox_lambda: float | None = None,
        seed_list: list[int] | None = None,
        display_result: bool = True,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Fits and tests the selected_model; evaluates competition score"""

    if boxcox_lambda is None:
        boxcox_lambda = _GLOBAL_LAMBDA

    def fit_data(train_x_, train_cons_y_, train_rate_y_):

        x_train, sc, consumed_svd, infra_svd, _cat_cols = preprocess_data(train_x_)
        _y_train = _get_modified_target(train_cons_y_, boxcox_lambda)

        models, pred_vals = [], []
        for _model in model_names:
            _one_models, _one_pred_vals = _modeling_with_some_seeds(_model, model_params, x_train, _y_train, boxcox_lambda, seed_list=seed_list, category_columns=_cat_cols)
            models.extend(_one_models)
            pred_vals.extend(_one_pred_vals)

        consumption = train_cons_y_.copy()
        consumption['cons_pred'] = np.mean(pred_vals, axis=0)

        pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        ir = model.fit_isotonic_regression(pred_rate_y, train_rate_y_)

        pred_rate_y = model.transform_isotonic_regression(pred_rate_y, ir)

        print('train comp score:', calc.weighted_average_of_consumption_and_poverty_rate(consumption, train_rate_y_, pred_rate_y))

        return models, pred_vals, sc, consumed_svd, infra_svd, ir, consumption, pred_rate_y, x_train


    def pred_data(test_x_, test_cons_y_, sc: StandardScaler, consumed_svd: TruncatedSVD, infra_svd: TruncatedSVD, models: list, ir: IsotonicRegression):

        x_test, *_ = preprocess_data(test_x_, sc, consumed_svd=consumed_svd, infra_svd=infra_svd)
        pred_cons_ys = _fitting_with_some_models(models, x_test, boxcox_lambda)

        pred_cons_y = np.mean(pred_cons_ys, axis=0)

        y_test = test_cons_y_.loc[:, 'cons_ppp17']

        consumption = test_cons_y_.copy()
        consumption['cons_pred'] = pred_cons_y

        pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        pred_rate_y = model.transform_isotonic_regression(pred_rate_y, ir)

        return x_test, y_test, consumption, pred_cons_y, pred_rate_y

    def calculate_metrics(pred_cons_y, y, pred_rate_y, consumption, models: list, X, target_rate_y) -> dict[str, float]:
        rmse = np.sqrt(np.mean((pred_cons_y - y) ** 2))
        mae = np.mean(np.abs(pred_cons_y - y))
        r2 = np.mean([_lb.score(X, y) for _lb in models])
        competition_score = calc.weighted_average_of_consumption_and_poverty_rate(consumption, pred_rate_y, target_rate_y)

        return dict(
            rmse=rmse,
            mae=mae,
            r2=r2,
            competition_score=competition_score
        )

    def show_metrics(scores_: list[dict[str, float]]):

        _scores_df = pd.DataFrame(scores_)
        print('\ntotal_score\n-----------')
        print(_scores_df.mean(axis=0))
        print('\nstd_score\n----------')
        print(_scores_df.std(axis=0))
        print('\n')
        print(_scores_df)

    _datas = file.get_datas()
    train_scores = []
    test_scores = []

    _k_fold_test_ids = [100000, 200000, 300000]

    for _i, _id in enumerate(_k_fold_test_ids):

        print(f'\nk-fold: {_i+1}/{len(_k_fold_test_ids)}: {_id}')

        train_x, train_cons_y, train_rate_y, test_x, test_cons_y, test_rate_y = data.split_datas(
            _datas['train'], _datas['target_consumption'], _datas['target_rate'], test_survey_ids=[_id]
        )

        _models, _pred_vals, _sc, _consumed_svd, _infra_svd, _ir, _consumption, _pred_rate_y, _x_train = fit_data(train_x, train_cons_y, train_rate_y)

        _train_metrics = calculate_metrics(_consumption['cons_pred'].to_numpy(), train_cons_y.loc[:, 'cons_ppp17'], _pred_rate_y, _consumption, _models, _x_train, train_rate_y)

        _x_test, _y_test, _consumption, _pred_cons_y, _pred_rate_y = pred_data(test_x, test_cons_y, _sc, _consumed_svd, _infra_svd, _models, _ir)

        _test_metrics = calculate_metrics(_pred_cons_y, _y_test, _pred_rate_y, _consumption, _models, _x_test, test_rate_y)

        _train_metrics['survey_ids'] = set(_k_fold_test_ids) - {_id}
        _test_metrics['survey_ids'] = _id

        train_scores.append(_train_metrics)
        test_scores.append(_test_metrics)

    if display_result:
        show_metrics(test_scores)

    return train_scores, test_scores


def fit_and_predictions_model(model_names: list[str], folder_prefix: str | None = None):
    """Fits the model; predicts consumption; saves the submission format"""

    _datas = file.get_datas()

    # learning
    _x_train, _sc, _consumed_svd, _infra_svd, _cat_cols = preprocess_data(_datas['train'])
    _y_train = _get_modified_target(_datas['target_consumption'])

    _models, pred_vals = [], []
    for _model in model_names:
        _one_models, _one_pred_vals = _modeling_with_some_seeds(model_name=_model, model_params=None, x_train=_x_train, y_train=_y_train, boxcox_lambda=_GLOBAL_LAMBDA, category_columns=_cat_cols)
        _models.extend(_one_models)
        pred_vals.extend(_one_pred_vals)

    _consumption = _datas['target_consumption']
    _consumption['cons_pred'] = np.mean(pred_vals, axis=0)
    pred_rate_y = calc.poverty_rates_from_consumption(_consumption, 'cons_pred')
    ir = model.fit_isotonic_regression(pred_rate_y, _datas['target_rate'])

    # prediction
    _predicted_coxbox = pred_models(_models, _sc)
    _predicted = calc.inverse_boxcox_transform(_predicted_coxbox, _GLOBAL_LAMBDA)

    _consumption, _ = file.get_submission_formats('../results')
    _consumption['cons_pred'] = _predicted
    pred_rate_y = calc.poverty_rates_from_consumption(_consumption, 'cons_pred')
    pred_rate_y = model.transform_isotonic_regression(pred_rate_y, ir)

    file.save_to_submission_format(_predicted, pred_rate=pred_rate_y, folder_prefix=folder_prefix)


def pred_models(fit_models: list, sc: StandardScaler) -> np.ndarray:
    _datas = file.get_datas()

    _datas_std, _ = preprocess.standardized_with_numbers_dataframe(_datas['test'], sc)
    _datas_category = preprocess.encoding_category_dataframe(_datas['test'])

    _datas = pd.concat([_datas_std, _datas_category], axis=1)


    return np.mean([_fit_model.predict(_datas) for _fit_model in fit_models], axis=0)


def preprocess_data(
        datas: pd.DataFrame,
        sc: StandardScaler | None = None,
        consumed_svd: TruncatedSVD | None = None,
        infra_svd: TruncatedSVD | None = None
) -> tuple[pd.DataFrame, StandardScaler, TruncatedSVD, TruncatedSVD, list[str]]:
    """Transforms input; returns normalized data and fitted objects"""

    _datas_category = preprocess.encoding_category_dataframe(datas)
    _datas_consumed, consumed_svd = preprocess.consumed_svd_dataframe(_datas_category, svd=consumed_svd)
    _datas_infrastructure, infra_svd = preprocess.infrastructure_svd_dataframe(_datas_category, svd=infra_svd)

    _new_datas = pd.concat([datas.copy().reset_index(drop=True), _datas_consumed.reset_index(drop=True), _datas_infrastructure.reset_index(drop=True)], axis=1)
    _datas_std, sc = preprocess.standardized_with_numbers_dataframe(_new_datas, sc, add_columns=_datas_consumed.columns.tolist() + _datas_infrastructure.columns.tolist())

    category_cols = _datas_category.columns

    return pd.concat([_datas_std.reset_index(drop=True), _datas_category.reset_index(drop=True)], axis=1), sc, consumed_svd, infra_svd, list(category_cols)


def _get_modified_target(targets: pd.DataFrame, boxcox_lambda: float | None = None) -> pd.DataFrame:

    if boxcox_lambda is None:
        boxcox_lambda = _GLOBAL_LAMBDA

    return calc.apply_boxcox_transform(targets.loc[:, 'cons_ppp17'], boxcox_lambda)[0]


def _modeling_with_some_seeds(model_name: str, model_params: dict | None, x_train, y_train, boxcox_lambda: float, seed_list: list[int] | None = None, category_columns: list[str] | None = None) -> tuple[list, list[np.ndarray]]:

    if seed_list is None:
        random.seed(0)
        _seeds_length = 2

        # seed_list = [123] + random.sample(range(1, 1000), _seeds_length)
        seed_list = [123]

    model_with_preds = [
        getattr(model, f'fit_{model_name}')(x_train, y_train, seed=_seed, params=model_params, categorical_cols=category_columns)
        if model_name == 'lightgbm' else
        getattr(model, f'fit_{model_name}')(x_train, y_train, seed=_seed, params=model_params)
        for _seed in tqdm(seed_list, desc=f'{model_name}: modeling with some seeds')
    ]
    models = [_model for _model, _ in model_with_preds]
    preds = [calc.inverse_boxcox_transform(_preds_boxcox, boxcox_lambda) for _, _preds_boxcox in model_with_preds]

    return models, preds


def _fitting_with_some_models(models, x_test, boxcox_lambda: float) -> list[np.ndarray]:
    _preds_boxcox = [_model.predict(x_test) for _model in models]

    return [calc.inverse_boxcox_transform(_pred_bc, boxcox_lambda) for _pred_bc in _preds_boxcox]

