__all__ = ['fit_and_test_pipeline', 'test_model_pipeline', 'fit_and_predictions_pipeline']


import numpy as np
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline

from .. import file, model, data, calc


def fit_and_test_pipeline() -> tuple[list[StackingRegressor], list[dict], list[dict]]:

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

    boxcox_lambda = 0.09
    _model_names = ['lightgbm', 'ridge', 'catboost', 'xgboost']

    stacking_regressor, model_pipelines = model.get_stacking_regressor_and_pipelines(_model_names, boxcox_lambda=boxcox_lambda)

    _datas = file.get_datas()

    _k_fold_test_ids = [100000, 200000, 300000]

    learned_stacks = []
    train_scores = []
    test_scores = []

    for _i, _id in enumerate(_k_fold_test_ids):
        print(f'\nk-fold: {_i + 1}/{len(_k_fold_test_ids)}: {_id}')

        train_x, train_cons_y, train_rate_y, test_x, test_cons_y, test_rate_y = data.split_datas(
            _datas['train'], _datas['target_consumption'], _datas['target_rate'], test_survey_ids=[_id]
        )

        set_config(transform_output="pandas")
        train_y = calc.apply_boxcox_transform(train_cons_y.cons_ppp17.to_numpy(), boxcox_lambda)[0]
        stacking_regressor.fit(train_x, train_y.astype(np.float32).flatten())

        # fitの後に実行
        model_names = [name for name, _ in model_pipelines]
        weights = get_feature_importance(stacking_regressor.final_estimator_)
        for name, weight in zip(model_names, weights):
            print(f"Model: {name}, Weight: {weight:.4f}")

        y_train_pred = stacking_regressor.predict(train_x)
        y_test_pred = stacking_regressor.predict(test_x)

        _y_train_mean_pred = calc.inverse_boxcox_transform(y_train_pred, boxcox_lambda)
        _y_test_mean_pred = calc.inverse_boxcox_transform(y_test_pred, boxcox_lambda)

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

        # plot_model_bias(_y_test_mean_pred, test_cons_y.cons_ppp17, "Stacking Regressor")

    return learned_stacks, train_scores, test_scores


def test_model_pipeline(model_name: str, model_params: dict | None = None) -> tuple[list[Pipeline], list[dict], list[dict]]:
    boxcox_lambda = 0.09

    _model_pipeline = model.get_stacking_regressor_and_pipelines(
        [model_name],
        boxcox_lambda=boxcox_lambda,
        model_params={model_name: model_params}
    )[1][0][1]

    _datas = file.get_datas()

    _k_fold_test_ids = [100000, 200000, 300000]

    learned_models = []
    train_scores = []
    test_scores = []

    for _i, _id in enumerate(_k_fold_test_ids):
        print(f'\nk-fold: {_i + 1}/{len(_k_fold_test_ids)}: {_id}')

        train_x, train_cons_y, train_rate_y, test_x, test_cons_y, test_rate_y = data.split_datas(
            _datas['train'], _datas['target_consumption'], _datas['target_rate'], test_survey_ids=[_id]
        )

        set_config(transform_output="pandas")
        train_y = calc.apply_boxcox_transform(train_cons_y.cons_ppp17.to_numpy(), boxcox_lambda)[0]
        _model_pipeline.fit(train_x, train_y.astype(np.float32).flatten())

        y_train_pred = _model_pipeline.predict(train_x)
        y_test_pred = _model_pipeline.predict(test_x)

        _y_train_mean_pred = calc.inverse_boxcox_transform(y_train_pred, boxcox_lambda)
        _y_test_mean_pred = calc.inverse_boxcox_transform(y_test_pred, boxcox_lambda)

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

    return learned_models, train_scores, test_scores,


def fit_and_predictions_pipeline(folder_prefix: str | None = None):
    boxcox_lambda = 0.

    stacking_regressor, model_pipelines = model.get_stacking_regressor_and_pipelines()

    _datas = file.get_datas()

    # learning
    train_x = _datas['train']
    train_cons_y = _datas['target_consumption']
    train_rate_y = _datas['target_rate']

    set_config(transform_output="pandas")
    train_y = calc.apply_boxcox_transform(train_cons_y.cons_ppp17.to_numpy(), boxcox_lambda)[0]
    stacking_regressor.fit(train_x, train_y.astype(np.float32).flatten())

    y_train_pred = stacking_regressor.predict(train_x)
    y_train_pred = calc.inverse_boxcox_transform(y_train_pred, boxcox_lambda)

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
    _predicted = calc.inverse_boxcox_transform(_predicted_coxbox, boxcox_lambda)

    _consumption, _ = file.get_submission_formats('../results')
    _consumption['cons_pred'] = _predicted
    pred_rate_y = calc.poverty_rates_from_consumption(_consumption, 'cons_pred')
    pred_rate_y = model.transform_isotonic_regression(pred_rate_y, ir)

    file.save_to_submission_format(_predicted, pred_rate=pred_rate_y, folder_prefix=folder_prefix)


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
