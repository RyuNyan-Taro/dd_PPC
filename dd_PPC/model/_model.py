__all__ = [
    'fit_random_forest',
    'fit_lightgbm',
    'fit_xgboost',
    'fit_catboost',
    'fit_isotonic_regression',
    'transform_isotonic_regression'
]

from typing import Any

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoost
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb

def fit_random_forest(x_train_std, y_train, show_fit_process: bool = True, seed: int = 42) -> tuple[RandomForestRegressor, np.ndarray]:
    _verbose = 2 if show_fit_process else 0

    RF = RandomForestRegressor(verbose=_verbose, random_state=seed, n_estimators=100)
    RF.fit(x_train_std, y_train)

    pred_RF = RF.predict(x_train_std)

    return RF, pred_RF


def fit_lightgbm(x_train, y_train, seed: int = 42, categorical_cols: list[str] = None, params: dict | None = None) -> tuple[lgb.LGBMRegressor, np.ndarray]:
    if params is None:
        params = dict(
            verbose=-1,
            n_estimators=3000,
            force_row_wise=True,
            bagging_fraction=0.8,
            bagging_freq=5
        )
    params['random_state'] = seed
    print(params)

    model = lgb.LGBMRegressor(**params)

    pred_y = model.fit(x_train, y_train, categorical_feature=categorical_cols if categorical_cols else 'auto')

    pred_lgb = pred_y.predict(x_train)

    return pred_y, pred_lgb


def fit_xgboost(x_train, y_train, seed: int = 42, params: dict | None = None) -> tuple[xgb.XGBRegressor, np.ndarray]:
    if params is None:
        params = dict(
            n_estimators=200,
            subsample=0.8
        )
    params['random_state'] = seed

    model = xgb.XGBRegressor(**params)

    pred_y = model.fit(x_train, y_train)

    pred_xgb = pred_y.predict(x_train)

    return pred_y, pred_xgb


def fit_catboost(x_train, y_train, seed: int = 42, params: dict | None = None) -> tuple[CatBoost, Any]:
    if params is None:
        params = dict(
            iterations=1100,
            learning_rate=0.03,
            depth=6,
            loss_function='RMSE',
            verbose=0
        )
    params['random_seed'] = seed

    model = catboost.CatBoostRegressor(**params)

    pred_y = model.fit(x_train, y_train)

    pred_cat = pred_y.predict(x_train)

    return pred_y, pred_cat


def fit_isotonic_regression(pred_rate: pd.DataFrame, target_rate: pd.DataFrame) -> IsotonicRegression:

    # drop survey_id and flatten
    X = pred_rate.T.to_numpy()[1:].flatten()
    y = target_rate.T.to_numpy()[1:].flatten()

    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(X, y)

    return ir


def transform_isotonic_regression(pred_rate: pd.DataFrame, ir: IsotonicRegression) -> pd.DataFrame:

    _datas = []
    for _id, _group in pred_rate.groupby('survey_id'):
        _rates = _group.T.to_numpy()[1:]

        _transformed = ir.transform(_rates.flatten()).T
        _datas.append(np.append(np.array(_id), _transformed))

    result_df = pd.DataFrame(
        data=_datas,
        index=pred_rate.index,
        columns=pred_rate.columns
    )
    result_df['survey_id'] = result_df['survey_id'].astype('int64')

    return result_df