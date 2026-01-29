__all__ = ['fit_and_test_pipeline', 'test_model_pipeline', 'fit_and_predictions_pipeline', 'sweep_quantile_alpha']


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.ensemble import StackingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from scipy.stats import boxcox_normmax

from .. import file, model, data, calc
from ..model import _nn as model_nn


_MODEL_NAMES = ['lgb_quantile', 'lgb_quantile_low', 'catboost', 'ridge', 'mlp_regressor']
_BOXCOX_LAMBDA = 0.09
_TARGET_TRANSFORM = dict(method='boxcox', quantile_n=1000)
_RATE_LINEAR_BLEND = 0
_RATE_Q_BLEND = 0.0
_RATE_Q_LOW_BLEND = 0.0
_RATE_DIST_BLEND = 0.2
_RATE_DIST_ALPHA = 1.0
_MTL_CONS_BLEND = 0.05
_MTL_RATE_BLEND = 0.15
_POVERTY_THRESHOLDS = np.array([
    3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.70, 8.40, 9.13,
    9.87, 10.70, 11.62, 12.69, 14.03, 15.64, 17.76, 20.99, 27.37
], dtype=np.float32)


def _fit_target_transform(y: np.ndarray, boxcox_lambda: float | None = None) -> tuple[np.ndarray, dict]:
    if boxcox_lambda is None:
        boxcox_lambda = _BOXCOX_LAMBDA

    return calc.fit_target_transform(
        y,
        method=_TARGET_TRANSFORM['method'],
        lambda_param=boxcox_lambda,
        quantile_n=_TARGET_TRANSFORM.get('quantile_n', 1000)
    )


def _normalize_rate_columns(df: pd.DataFrame, reference_cols: pd.Index) -> pd.DataFrame:
    rate_cols = [c for c in reference_cols if c != 'survey_id']

    def _to_float(col: str) -> float | None:
        if col == 'survey_id':
            return None
        try:
            return float(col.replace('pct_hh_below_', ''))
        except ValueError:
            return None

    ref_map = {}
    for col in rate_cols:
        val = _to_float(col)
        if val is not None:
            ref_map[round(val, 3)] = col

    rename_map = {}
    for col in df.columns:
        val = _to_float(col)
        if val is None:
            continue
        key = round(val, 3)
        if key in ref_map:
            rename_map[col] = ref_map[key]

    renamed = df.rename(columns=rename_map)

    # ensure all reference columns exist
    aligned = pd.DataFrame(columns=reference_cols)
    aligned['survey_id'] = renamed['survey_id'] if 'survey_id' in renamed.columns else df.get('survey_id')
    for col in rate_cols:
        if col in renamed.columns:
            aligned[col] = renamed[col]
        elif col in df.columns:
            aligned[col] = df[col]
        else:
            aligned[col] = np.nan

    return aligned


def _blend_poverty_rates(base_rates: pd.DataFrame, extra_rates: list[tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    rate_cols = [c for c in base_rates.columns if c != 'survey_id']
    base_vals = base_rates[rate_cols].to_numpy()
    total_weight = max(0.0, 1.0 - sum(w for _, w in extra_rates))
    blended_vals = base_vals * total_weight

    for rates_df, weight in extra_rates:
        if weight <= 0:
            continue
        aligned = _normalize_rate_columns(rates_df, base_rates.columns)
        blended_vals += aligned[rate_cols].to_numpy() * weight
        total_weight += weight

    if total_weight == 0:
        return base_rates

    blended = base_rates.copy()
    blended[rate_cols] = blended_vals / total_weight
    return blended


def _build_poverty_targets(consumption: pd.Series) -> np.ndarray:
    cons_arr = consumption.to_numpy().astype(np.float32)
    return (cons_arr[:, None] < _POVERTY_THRESHOLDS[None, :]).astype(np.float32)


def _pov_probs_to_rates(pov_probs: np.ndarray, survey_ids: pd.Series) -> pd.DataFrame:
    rate_cols = [f'pct_hh_below_{t}' for t in _POVERTY_THRESHOLDS]
    df = pd.DataFrame(pov_probs, columns=rate_cols)
    df['survey_id'] = survey_ids.to_numpy()
    grouped = df.groupby('survey_id')
    rates = grouped[rate_cols].mean().reset_index()
    return rates


def _get_poverty_weights() -> torch.Tensor:
    weights = [1 - abs(0.4 - p / 100) for p in range(5, 100, 5)]
    return torch.tensor(weights, dtype=torch.float32)


def _train_mtl_model(X: np.ndarray, cons_target: np.ndarray, pov_target: np.ndarray, epochs: int = 5, batch_size: int = 256,
                    lr: float = 1e-3, seed: int = 123) -> tuple[nn.Module, str]:
    # fix seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    input_dim = X.shape[1]
    model_mtl = model_nn.MTLConsPoverty(input_dim=input_dim, pov_dim=pov_target.shape[1]).to(device)
    criterion = model_nn.MTLLossWrapper(_get_poverty_weights().to(device))
    optimizer = torch.optim.Adam(model_mtl.parameters(), lr=lr)

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(cons_target, dtype=torch.float32),
        torch.tensor(pov_target, dtype=torch.float32)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model_mtl.train()
    for _ in range(epochs):
        for xb, yb_cons, yb_pov in loader:
            xb = xb.to(device)
            yb_cons = yb_cons.to(device)
            yb_pov = yb_pov.to(device)

            optimizer.zero_grad()
            cons_pred, pov_pred = model_mtl(xb)
            loss = criterion((cons_pred, pov_pred), (yb_cons, yb_pov))
            loss.backward()
            optimizer.step()

    return model_mtl, device


def _predict_mtl_model(model_mtl: nn.Module, device: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model_mtl.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32).to(device)
        cons_pred, pov_pred = model_mtl(xb)
    return cons_pred.cpu().numpy(), pov_pred.cpu().numpy()


def fit_and_test_pipeline() -> tuple[list[StackingRegressor], list[dict], list[dict], list[IsotonicRegression]]:

    def get_feature_importance(model_):
        """Extract feature importance/coefficients based on the model type."""
        if hasattr(model_, 'coef_'):
            # Linear models (Ridge, Lasso, LinearRegression, etc.)
            return model_.coef_
        elif hasattr(model_, 'feature_importances_'):
            # Tree-based models (LightGBM, XGBoost, CatBoost, RandomForest, etc.)
            return model_.feature_importances_
        else:
            raise AttributeError(f"Model {type(model_).__name__} doesn't have coef_ or feature_importances_")

    boxcox_lambda = _BOXCOX_LAMBDA
    _model_names = _MODEL_NAMES

    _datas = file.get_datas()

    _k_fold_test_ids = [100000, 200000, 300000]

    learned_stacks = []
    train_scores = []
    test_scores = []
    isotonic_regressors = []

    for _i, _id in enumerate(_k_fold_test_ids):
        print(f'\nk-fold: {_i + 1}/{len(_k_fold_test_ids)}: {_id}')

        train_x, train_cons_y, train_rate_y, test_x, test_cons_y, test_rate_y = data.split_datas(
            _datas['train'], _datas['target_consumption'], _datas['target_rate'], test_survey_ids=[_id]
        )

        set_config(transform_output="pandas")
        train_y, target_transform_state = _fit_target_transform(train_cons_y.cons_ppp17.to_numpy())
        stacking_regressor, model_pipelines = model.get_stacking_regressor_and_pipelines(
            _model_names,
            boxcox_lambda=boxcox_lambda,
            target_transform_state=target_transform_state
        )

        _set_group_cv_splits(stacking_regressor, train_x, train_y, train_x['survey_id'])
        stacking_regressor.fit(
            train_x,
            train_y.astype(np.float32).flatten()
        )

        # fitの後に実行
        model_names = [name for name, _ in model_pipelines]
        weights = get_feature_importance(stacking_regressor.final_estimator_)
        for name, weight in zip(model_names, weights):
            print(f"Model: {name}, Weight: {weight:.4f}")

        y_train_pred = stacking_regressor.predict(train_x)
        y_test_pred = stacking_regressor.predict(test_x)

        _y_train_mean_pred = calc.inverse_target_transform(y_train_pred, target_transform_state)
        _y_test_mean_pred = calc.inverse_target_transform(y_test_pred, target_transform_state)

        # # MTL branch
        # prep = model_pipelines[0][1].named_steps['prep']
        # prep_fitted = prep.fit(train_x)
        # train_feat = prep_fitted.transform(train_x)
        # test_feat = prep_fitted.transform(test_x)
        # if hasattr(train_feat, "to_numpy"):
        #     train_feat = train_feat.to_numpy()
        #     test_feat = test_feat.to_numpy()
        # pov_target = _build_poverty_targets(train_cons_y.cons_ppp17)
        # mtl_model, mtl_device = _train_mtl_model(
        #     train_feat,
        #     train_cons_y.cons_ppp17.to_numpy().astype(np.float32),
        #     pov_target
        # )
        # mtl_cons_train, mtl_pov_train = _predict_mtl_model(mtl_model, mtl_device, train_feat)
        # mtl_cons_test, mtl_pov_test = _predict_mtl_model(mtl_model, mtl_device, test_feat)
        #
        # _y_train_mean_pred = (1 - _MTL_CONS_BLEND) * _y_train_mean_pred + _MTL_CONS_BLEND * mtl_cons_train
        # _y_test_mean_pred = (1 - _MTL_CONS_BLEND) * _y_test_mean_pred + _MTL_CONS_BLEND * mtl_cons_test

        consumption = train_cons_y.copy()
        consumption['cons_pred'] = _y_train_mean_pred
        train_pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')
        # mtl_train_rate = _pov_probs_to_rates(mtl_pov_train, consumption['survey_id'])
        # if _MTL_RATE_BLEND > 0:
        #     train_pred_rate_y = _blend_poverty_rates(train_pred_rate_y, [(mtl_train_rate, _MTL_RATE_BLEND)])

        ir = model.fit_isotonic_regression(train_pred_rate_y, train_rate_y)

        train_pred_rate_y = model.transform_isotonic_regression(train_pred_rate_y, ir)

        _train_metrics = _calculate_metrics(_y_train_mean_pred, train_cons_y.cons_ppp17, train_pred_rate_y, consumption, train_rate_y)

        consumption = test_cons_y.copy()
        consumption['cons_pred'] = _y_test_mean_pred
        test_pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')
        # mtl_test_rate = _pov_probs_to_rates(mtl_pov_test, consumption['survey_id'])
        # if _MTL_RATE_BLEND > 0:
        #     test_pred_rate_y = _blend_poverty_rates(test_pred_rate_y, [(mtl_test_rate, _MTL_RATE_BLEND)])

        test_pred_rate_y = model.transform_isotonic_regression(test_pred_rate_y, ir)

        _test_metrics = _calculate_metrics(_y_test_mean_pred, test_cons_y.cons_ppp17, test_pred_rate_y, consumption, test_rate_y)

        print(_train_metrics)
        print(_test_metrics)

        learned_stacks.append(stacking_regressor)
        train_scores.append(_train_metrics)
        test_scores.append(_test_metrics)
        isotonic_regressors.append(ir)

        # plot_model_bias(_y_test_mean_pred, test_cons_y.cons_ppp17, "Stacking Regressor")

    return learned_stacks, train_scores, test_scores, isotonic_regressors


def test_model_pipeline(model_name: str, model_params: dict | None = None) -> tuple[list[Pipeline], list[dict], list[dict], list[IsotonicRegression]]:
    """Tests a machine learning pipeline for a given model name and optional parameters using k-fold cross-validation.
    The function builds and trains a stacking regressor pipeline using the input model, processes data using box-cox
    transformations, and evaluates the pipeline's performance via metrics such as consumption and poverty rate predictions.
    Models, training scores, and testing scores are returned for further analysis.

    Args:
        model_name (str): The name of the model to be used in the stacking regressor pipeline.
        model_params (dict | None): A dictionary of model parameters, or None if no parameters are specified.

    Returns:
        A tuple containing:
            - A list of trained machine learning Pipeline objects.
            - A list of dictionaries containing training evaluation metrics.
            - A list of dictionaries containing testing evaluation metrics.
    """
    boxcox_lambda = _BOXCOX_LAMBDA

    _datas = file.get_datas()

    _k_fold_test_ids = [100000, 200000, 300000]

    learned_models = []
    train_scores = []
    test_scores = []
    isotonic_regressors = []

    for _i, _id in enumerate(_k_fold_test_ids):
        print(f'\nk-fold: {_i + 1}/{len(_k_fold_test_ids)}: {_id}')

        train_x, train_cons_y, train_rate_y, test_x, test_cons_y, test_rate_y = data.split_datas(
            _datas['train'], _datas['target_consumption'], _datas['target_rate'], test_survey_ids=[_id]
        )

        set_config(transform_output="pandas")
        train_y, target_transform_state = _fit_target_transform(train_cons_y.cons_ppp17.to_numpy())
        _model_pipeline = model.get_stacking_regressor_and_pipelines(
            [model_name],
            boxcox_lambda=boxcox_lambda,
            model_params=model_params,
            target_transform_state=target_transform_state
        )[1][0][1]

        _model_pipeline.fit(train_x, train_y.astype(np.float32).flatten())

        y_train_pred = _model_pipeline.predict(train_x)
        y_test_pred = _model_pipeline.predict(test_x)

        _y_train_mean_pred = calc.inverse_target_transform(y_train_pred, target_transform_state)
        _y_test_mean_pred = calc.inverse_target_transform(y_test_pred, target_transform_state)

        consumption = train_cons_y.copy()
        consumption['cons_pred'] = _y_train_mean_pred
        train_pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        ir = model.fit_isotonic_regression(train_pred_rate_y, train_rate_y)

        train_pred_rate_y = model.transform_isotonic_regression(train_pred_rate_y, ir)

        _train_metrics = _calculate_metrics(_y_train_mean_pred, train_cons_y.cons_ppp17, train_pred_rate_y, consumption,
                                            train_rate_y)

        consumption = test_cons_y.copy()
        consumption['cons_pred'] = _y_test_mean_pred
        test_pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        test_pred_rate_y = model.transform_isotonic_regression(test_pred_rate_y, ir)

        _test_metrics = _calculate_metrics(_y_test_mean_pred, test_cons_y.cons_ppp17, test_pred_rate_y, consumption,
                                           test_rate_y)

        print(_train_metrics)
        print(_test_metrics)

        learned_models.append(_model_pipeline)
        train_scores.append(_train_metrics)
        test_scores.append(_test_metrics)
        isotonic_regressors.append(ir)

    return learned_models, train_scores, test_scores, isotonic_regressors


def fit_and_predictions_pipeline(folder_prefix: str | None = None):

    boxcox_lambda = _BOXCOX_LAMBDA

    _datas = file.get_datas()

    # learning
    train_x = _datas['train']
    train_cons_y = _datas['target_consumption']
    train_rate_y = _datas['target_rate']

    set_config(transform_output="pandas")
    train_y, target_transform_state = _fit_target_transform(train_cons_y.cons_ppp17.to_numpy())
    stacking_regressor, model_pipelines = model.get_stacking_regressor_and_pipelines(
        _MODEL_NAMES,
        boxcox_lambda=boxcox_lambda,
        target_transform_state=target_transform_state
    )

    _set_group_cv_splits(stacking_regressor, train_x, train_y, train_x['survey_id'], n_splits=3)
    stacking_regressor.fit(
        train_x,
        train_y.astype(np.float32).flatten()
    )

    y_train_pred = stacking_regressor.predict(train_x)
    y_train_pred = calc.inverse_target_transform(y_train_pred, target_transform_state)

    # MTL branch
    # prep = model_pipelines[0][1].named_steps['prep']
    # prep_fitted = prep.fit(train_x)
    # train_feat = prep_fitted.transform(train_x)
    # test_feat = prep_fitted.transform(_datas['test'])
    # if hasattr(train_feat, "to_numpy"):
    #     train_feat = train_feat.to_numpy()
    #     test_feat = test_feat.to_numpy()
    # pov_target = _build_poverty_targets(train_cons_y.cons_ppp17)
    # mtl_model, mtl_device = _train_mtl_model(
    #     train_feat,
    #     train_cons_y.cons_ppp17.to_numpy().astype(np.float32),
    #     pov_target
    # )
    # mtl_cons_train, mtl_pov_train = _predict_mtl_model(mtl_model, mtl_device, train_feat)
    # mtl_cons_test, mtl_pov_test = _predict_mtl_model(mtl_model, mtl_device, test_feat)
    #
    # y_train_pred = (1 - _MTL_CONS_BLEND) * y_train_pred + _MTL_CONS_BLEND * mtl_cons_train

    train_cons_y['cons_pred'] = y_train_pred
    train_pred_rate_y = calc.poverty_rates_from_consumption(train_cons_y, 'cons_pred')
    # mtl_train_rate = _pov_probs_to_rates(mtl_pov_train, train_cons_y['survey_id'])
    # if _MTL_RATE_BLEND > 0:
    #     train_pred_rate_y = _blend_poverty_rates(train_pred_rate_y, [(mtl_train_rate, _MTL_RATE_BLEND)])
    ir = model.fit_isotonic_regression(train_pred_rate_y, train_rate_y)
    train_pred_rate_y = model.transform_isotonic_regression(train_pred_rate_y, ir)

    print(_calculate_metrics(
        y_train_pred, train_cons_y.cons_ppp17, train_pred_rate_y, train_cons_y,
        train_rate_y
    ))

    # prediction
    _predicted_coxbox = stacking_regressor.predict(_datas['test'])
    _predicted = calc.inverse_target_transform(_predicted_coxbox, target_transform_state)
    # _predicted = (1 - _MTL_CONS_BLEND) * _predicted + _MTL_CONS_BLEND * mtl_cons_test

    _consumption, _ = file.get_submission_formats('../results')
    _consumption['cons_pred'] = _predicted
    pred_rate_y = calc.poverty_rates_from_consumption(_consumption, 'cons_pred')
    # mtl_pred_rate_y = _pov_probs_to_rates(mtl_pov_test, _consumption['survey_id'])
    # if _MTL_RATE_BLEND > 0:
    #     pred_rate_y = _blend_poverty_rates(pred_rate_y, [(mtl_pred_rate_y, _MTL_RATE_BLEND)])
    pred_rate_y = model.transform_isotonic_regression(pred_rate_y, ir)

    file.save_to_submission_format(_predicted, pred_rate=pred_rate_y, folder_prefix=folder_prefix)


def sweep_quantile_alpha(alphas: list[float]) -> dict[float, tuple[list[dict], list[dict]]]:
    results = {}
    base_params = file.load_best_params('lightgbm')
    base_params['objective'] = 'quantile'
    base_params['metric'] = 'quantile'
    base_params['verbose'] = -1
    base_params['random_state'] = 123

    for alpha in alphas:
        params = base_params.copy()
        params['alpha'] = alpha
        _, train_scores, test_scores, _ = test_model_pipeline('lgb_quantile', model_params={'lgb_quantile': params})
        results[alpha] = (train_scores, test_scores)

    return results


# common subfunctions for pipelines
def _calculate_metrics(pred_cons_y, y, pred_rate_y, consumption, target_rate_y) -> dict[str, float]:
    rmse = np.sqrt(np.mean((pred_cons_y - y) ** 2))
    mae = np.mean(np.abs(pred_cons_y - y))
    competition_score = calc.weighted_average_of_consumption_and_poverty_rate(consumption, pred_rate_y, target_rate_y)

    return dict(
        rmse=rmse,
        mae=mae,
        competition_score=competition_score
    )


def _set_group_cv_splits(estimator: StackingRegressor, X, y, groups, n_splits: int = 2) -> None:
    """Precompute group-aware CV splits to avoid passing groups through fit()."""
    splitter = GroupKFold(n_splits=n_splits)
    estimator.cv = list(splitter.split(X, y, groups=groups))
