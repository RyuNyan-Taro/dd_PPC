__all__ = ['fit_and_test_pipeline', 'test_model_pipeline', 'fit_and_predictions_pipeline', 'sweep_quantile_alpha']


import numpy as np
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.ensemble import StackingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from .. import file, model, data, calc

_MODEL_NAMES = ['lightgbm', 'lgb_quantile', 'lgb_quantile_low', 'lgb_quantile_mid', 'catboost', 'ridge']
_BOXCOX_LAMBDA = 0.09
_TARGET_TRANSFORM = dict(method='boxcox', boxcox_lambda=_BOXCOX_LAMBDA, quantile_n=1000)


def _fit_target_transform(y: np.ndarray) -> tuple[np.ndarray, dict]:
    return calc.fit_target_transform(
        y,
        method=_TARGET_TRANSFORM['method'],
        lambda_param=_TARGET_TRANSFORM.get('boxcox_lambda'),
        quantile_n=_TARGET_TRANSFORM.get('quantile_n', 1000)
    )


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

        consumption = train_cons_y.copy()
        consumption['cons_pred'] = _y_train_mean_pred
        train_pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        ir = model.fit_isotonic_regression(train_pred_rate_y, train_rate_y)

        train_pred_rate_y = model.transform_isotonic_regression(train_pred_rate_y, ir)

        _train_metrics = _calculate_metrics(_y_train_mean_pred, train_cons_y.cons_ppp17, train_pred_rate_y, consumption, train_rate_y)

        consumption = test_cons_y.copy()
        consumption['cons_pred'] = _y_test_mean_pred
        test_pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

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

    _set_group_cv_splits(stacking_regressor, train_x, train_y, train_x['survey_id'])
    stacking_regressor.fit(
        train_x,
        train_y.astype(np.float32).flatten()
    )

    y_train_pred = stacking_regressor.predict(train_x)
    y_train_pred = calc.inverse_target_transform(y_train_pred, target_transform_state)

    train_cons_y['cons_pred'] = y_train_pred
    train_pred_rate_y = calc.poverty_rates_from_consumption(train_cons_y, 'cons_pred')
    ir = model.fit_isotonic_regression(train_pred_rate_y, train_rate_y)
    train_pred_rate_y = model.transform_isotonic_regression(train_pred_rate_y, ir)

    print(_calculate_metrics(
        y_train_pred, train_cons_y.cons_ppp17, train_pred_rate_y, train_cons_y,
        train_rate_y
    ))

    # prediction
    _predicted_coxbox = stacking_regressor.predict(_datas['test'])
    _predicted = calc.inverse_target_transform(_predicted_coxbox, target_transform_state)

    _consumption, _ = file.get_submission_formats('../results')
    _consumption['cons_pred'] = _predicted
    pred_rate_y = calc.poverty_rates_from_consumption(_consumption, 'cons_pred')
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
