__all__ = ['get_stacking_regressor_and_pipelines']

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, QuantileRegressor, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer, TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.impute import SimpleImputer
from category_encoders import CountEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost

from .. import file, model, calc
from .._config import CATEGORY_NUMBER_MAPS, NUMBER_COLUMNS
from .. import preprocess
from ..preprocess import complex_numbers_dataframe, survey_related_features

def _augment_meta_features(X):
    """Add simple stats over base predictions to meta features."""
    X_arr = np.asarray(X)
    stats = np.stack([
        X_arr.mean(axis=1),
        X_arr.std(axis=1),
        X_arr.min(axis=1),
        X_arr.max(axis=1),
        np.quantile(X_arr, 0.25, axis=1),
        np.quantile(X_arr, 0.75, axis=1),
    ], axis=1)
    return np.concatenate([X_arr, stats], axis=1)

_HUBER_VARIANT = 'v3'
_HUBER_PARAMS = {
    'base': dict(max_iter=10000, epsilon=1.1),
    'v2': dict(max_iter=20000, epsilon=1.2, alpha=0.0001),
    'v3': dict(max_iter=20000, epsilon=1.35, alpha=0.0005),
    'v4': dict(max_iter=20000, epsilon=1.4, alpha=0.0005),
    'v5': dict(max_iter=20000, epsilon=1.3, alpha=0.001),
}


def get_stacking_regressor_and_pipelines(
        model_names: list[str],
        boxcox_lambda: float,
        model_params: dict | None = None,
        target_transform_state: dict | None = None
) -> tuple[StackingRegressor, list[tuple[str, Pipeline]]]:
    """
    Constructs a stacking regressor and associated pipelines for given models by applying
    preprocessing and encapsulating them in pipelines.

    This function prepares a setup for stacking regression by preprocessing features like numeric
    and categorical columns, initializes the models using the provided or default parameters,
    and combines them into a `StackingRegressor`. Each model is wrapped in a `Pipeline` with
    a shared preprocessing step that handles transformations.

    Args:
        model_names: A list of model names to include in the stacking regression.
            Each model will be initialized and processed based on its specified parameters.
        boxcox_lambda: The lambda value for Box-Cox transformations, used when
            initializing certain models.
        model_params: Optional dictionary containing model-specific parameters.
            If None, default parameters will be retrieved automatically for each model.

    Returns:
        The first element is the `StackingRegressor` configured with the specified models and
        a final estimator.
        The second element is a list of tuples where each tuple contains the model name and
        its corresponding `Pipeline`.
    """

    num_cols, category_cols, complex_category_cols, category_number_maps = _get_columns()

    if model_params is None:
        model_params = _get_model_params(model_names)

    for _model, _params in model_params.items():
        print('model:', _model)
        print('params:', _params)

    preprocessor = _get_common_preprocess(category_number_maps, category_cols, num_cols)

    model_pipelines = [
        (
            _name,
            Pipeline([('prep', preprocessor)] + _get_initialized_model(
                _name,
                model_params,
                category_cols=complex_category_cols,
                boxcox_lambda=boxcox_lambda,
                target_transform_state=target_transform_state
            ))
        ) for _name in model_names]

    kf = GroupKFold(n_splits=2)

    stacking_regressor = StackingRegressor(
        estimators=model_pipelines,
        # final_estimator=Ridge(random_state=123, max_iter=10000, positive=True, alpha=1, fit_intercept=True),
        final_estimator=HuberRegressor(**_HUBER_PARAMS[_HUBER_VARIANT]),
        # final_estimator=lgb.LGBMRegressor(
        #     n_estimators=300,
        #     learning_rate=0.05,
        #     num_leaves=31,
        #     min_child_samples=50,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     reg_alpha=0.0,
        #     reg_lambda=1.0,
        #     random_state=123
        # ),
        # final_estimator=Lasso(**model_params['lasso']),
        # final_estimator=QuantileRegressor(quantile=0.5),
        cv=kf,
        n_jobs=1,
        verbose=1
    )

    return stacking_regressor, model_pipelines


# sub functions to get params
def _get_columns() -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    _access_or_not = {'Access': 1, 'No access': 0}
    _already_number = {0: 0, 1: 1}
    _yes_no = {'Yes': 1, 'No': 0}

    category_cols = [
        'water', 'toilet', 'sewer', 'elect', 'water_source',
        'male', 'urban',
        'owner', 'employed', 'any_nonagric',
        'sanitation_source', 'dweltyp', 'educ_max', 'sector1d',
        'region1', 'region2', 'region3', 'region4', 'region5', 'region6', 'region7',
        'consumed100', 'consumed200', 'consumed300', 'consumed400',
        'consumed500', 'consumed600', 'consumed700', 'consumed800',
        'consumed900', 'consumed1000', 'consumed1100', 'consumed1200',
        'consumed1300', 'consumed1400', 'consumed1500', 'consumed1600',
        'consumed1700', 'consumed1800', 'consumed1900', 'consumed2000',
        'consumed2100', 'consumed2200', 'consumed2300', 'consumed2400',
        'consumed2500', 'consumed2600', 'consumed2700', 'consumed2800',
        'consumed2900', 'consumed3000', 'consumed3100', 'consumed3200',
        'consumed3300', 'consumed3400', 'consumed3500', 'consumed3600',
        'consumed3700', 'consumed3800', 'consumed3900', 'consumed4000',
        'consumed4100', 'consumed4200', 'consumed4300', 'consumed4400',
        'consumed4500', 'consumed4600', 'consumed4700', 'consumed4800',
        'consumed4900', 'consumed5000',
    ]

    num_cols = NUMBER_COLUMNS.copy()

    _complex_input_cols = num_cols + category_cols + ['svd_consumed_0', 'svd_infrastructure_0', 'svd_consumed_1']

    complex_output_cols = list(complex_numbers_dataframe(pd.DataFrame(
        [[0] * len(_complex_input_cols), [1] * len(_complex_input_cols)],
        columns=_complex_input_cols)).columns)

    complex_category_output_cols = list(preprocess.complex_category_dataframe(pd.DataFrame(
        [[0] * len(_complex_input_cols), [1] * len(_complex_input_cols)],
        columns=_complex_input_cols)).columns)

    svd_cols = [f'svd_consumed_{i}' for i in range(3)] + [f'svd_infrastructure_{i}' for i in range(3)]

    _survey_cols = list({'survey_id', 'sanitation_source'} | set(num_cols) | {'educ_max'} | set(complex_output_cols) | set(svd_cols))

    survey_related_output_cols = list(survey_related_features(pd.DataFrame(
        [[0] * len(_survey_cols), [1] * len(_survey_cols)],
        columns=_survey_cols)).columns)

    final_num_cols = num_cols + svd_cols + complex_output_cols + survey_related_output_cols

    return final_num_cols, category_cols, complex_category_output_cols, CATEGORY_NUMBER_MAPS


def _get_model_params(model_names: list[str]) -> dict[str, dict]:
    model_params = {}
    for _model in ['lightgbm', 'lgb_quantile', 'lgb_quantile_low', 'lgb_quantile_mid', 'xgboost', 'catboost', 'ridge', 'lasso', 'elasticnet', 'kneighbors']:
        _model_param = file.load_best_params('lightgbm' if _model in ['lgb_quantile', 'lgb_quantile_low', 'lgb_quantile_mid'] else _model)
        match _model:
            case 'lightgbm':
                _model_param['objective'] = 'regression'
                _model_param['metric'] = 'rmse'
                _model_param['verbose'] = -1
            case 'lgb_quantile':
                _model_param['objective'] = 'quantile'
                _model_param['metric'] = 'quantile'
                _model_param['alpha'] = 0.3
                _model_param['verbose'] = -1
            case 'lgb_quantile_low':
                _model_param['objective'] = 'quantile'
                _model_param['metric'] = 'quantile'
                _model_param['alpha'] = 0.15
                _model_param['verbose'] = -1
            case 'lgb_quantile_mid':
                _model_param['objective'] = 'quantile'
                _model_param['metric'] = 'quantile'
                _model_param['alpha'] = 0.5
                _model_param['verbose'] = -1
            case 'xgboost':
                _model_param['objective'] = 'reg:squarederror'
                _model_param['eval_metric'] = 'rmse'
            case 'catboost':
                _model_param['verbose'] = 0
                _model_param['loss_function'] = 'RMSE'
            case 'lasso':
                _model_param['max_iter'] = 10000
            case 'ridge':
                _model_param['max_iter'] = 10000
            case 'elasticnet':
                _model_param['max_iter'] = 10000

        if _model != 'kneighbors':
            _model_param['random_state'] = 123
        model_params[_model] = _model_param

    model_params['tabular'] = dict(lr=0.01, max_epochs=4, batch_size=32, seed=123)
    model_params['mlp'] = dict(lr=0.001, max_epochs=7, batch_size=32, seed=123)

    for _threshold in ['clf_low', 'clf_middle', 'clf_high', 'clf_very_high']:
        model_params[_threshold] = dict(random_state=123, verbose=-1, force_row_wise=True)

    return {model: model_params[model] for model in model_names}


def _get_common_preprocess(category_number_maps: dict, category_cols: list[str], num_cols: list[str]) -> Pipeline:
    _category_stage = ColumnTransformer(
        transformers=[
            ('cat_encode', _CustomCategoryMapper(category_number_maps, category_cols), category_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    preprocessor = Pipeline([
        ('drop_unused', FunctionTransformer(_drop_unused_columns)),

        ('handle_null_numbers', FunctionTransformer(_handle_null_numbers)),

        ('category_encoding', _category_stage),

        ('svd_gen', _SVDFeatureGenerator()),

        ('complex_gen', FunctionTransformer(_complex_feature_wrapper)),

        ('survey_related_gen', FunctionTransformer(_survey_related_feature_wrapper)),

        ('drop_temporary_used', FunctionTransformer(_drop_temporary_used_columns)),

        ('final_scaler', ColumnTransformer(
            transformers=[
                ('scaling', StandardScaler(), num_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ))
    ])

    return preprocessor


def _get_initialized_model(
        model_name: str,
        model_params: dict,
        category_cols: list[str],
        boxcox_lambda: float,
        target_transform_state: dict | None
) -> list[tuple[str, BaseEstimator]]:
    _add_float_size_conversion = ['tabular', 'mlp']
    _add_count_encoding = ['lightgbm', 'lgb_quantile', 'lgb_quantile_low', 'lgb_quantile_mid', 'catboost', 'xgboost']
    _clf_model = ['clf_low', 'clf_middle', 'clf_high', 'clf_very_high']
    _model = None

    def get_transformed_threshold(original_val):
        if target_transform_state is None:
            return calc.apply_boxcox_transform(np.array([original_val]), boxcox_lambda)[0][0]
        return calc.transform_target_thresholds(np.array([original_val]), target_transform_state)[0]

    if model_name in _add_float_size_conversion:
        match model_name:
            case 'tabular':
                _model = model.get_tabular_nn_regressor(model_params['tabular'])
            case 'mlp':
                _model = model.get_mlp_nn_regressor(model_params['mlp'])
            case _:
                raise ValueError(f'Invalid model name: {model_name}')

        return [('convert', model.Float32Transformer()), _model]

    if model_name in _add_count_encoding:
        _model_dict = {
            'lightgbm': {'model': lgb.LGBMRegressor, 'drop': [
                'exp_per_hsize', 'any_nonagoric_and_sewer', 'has_child', 'urban_sanitation',
                'stable_workers', 'dependency_interaction'
            ]},
            'lgb_quantile': {'model': lgb.LGBMRegressor, 'drop': [
                'exp_per_hsize', 'any_nonagoric_and_sewer', 'has_child', 'urban_sanitation',
                'stable_workers', 'dependency_interaction'
            ]},
            'lgb_quantile_low': {'model': lgb.LGBMRegressor, 'drop': [
                'exp_per_hsize', 'any_nonagoric_and_sewer', 'has_child', 'urban_sanitation',
                'stable_workers', 'dependency_interaction'
            ]},
            'lgb_quantile_mid': {'model': lgb.LGBMRegressor, 'drop': [
                'exp_per_hsize', 'any_nonagoric_and_sewer', 'has_child', 'urban_sanitation',
                'stable_workers', 'dependency_interaction'
            ]},
            'catboost': {'model': catboost.CatBoostRegressor, 'drop': [
                'water', 'sewer', 'urban', 'has_child', 'stable_workers',
                'hsize_diff_survey', 'hsize_ratio_survey', 'hsize_rank_survey',
                'svd_complex_0', 'svd_complex_1', 'svd_complex_2', 'cat_head_profile'
            ]},
            'xgboost': {'model': xgb.XGBRegressor, 'drop': [
                'exp_per_hsize', 'lower_than_not_have_consumed', 'stable_workers',
                'hsize_diff_survey', 'hsize_ratio_survey', 'hsize_rank_survey', 'diff_consumed_to_strata',
                'dependency_interaction', 'svd_complex_0', 'svd_complex_1', 'svd_complex_2', 'cat_head_profile'
            ]}
        }[model_name]

        _convert_category_cols = ['educ_max']

        if model_name == 'catboost':
            model_params[model_name]['cat_features'] = list(set(category_cols) - set(_model_dict['drop']))
        elif model_name == 'xgboost':
            model_params[model_name]['enable_categorical'] = True

        _model = _model_dict['model'](**model_params[model_name])
        if model_name in ['lightgbm', 'lgb_quantile', 'lgb_quantile_low', 'lgb_quantile_mid']:
            _model.categorical_features_ = category_cols + _convert_category_cols

        _count_encoding_cols = ['sector1d']
        _ce = ColumnTransformer(
            transformers=[(
                'encoding', CountEncoder(cols=_count_encoding_cols, handle_unknown=0, normalize=True), _count_encoding_cols)],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        def _convert_category(X):

            df_cat = X[_convert_category_cols].copy()
            df_cat.index = X.index
            for _col in _convert_category_cols:
                df_cat[_col] = df_cat[_col].astype('category')

            return pd.concat([X.drop(columns=_convert_category_cols), df_cat], axis=1)

        def _drop_features(X):
            return X.drop(columns=_model_dict['drop'])

        if model_name in ['lightgbm', 'lgb_quantile', 'lgb_quantile_low', 'lgb_quantile_mid']:
            return [
                ('count_encoding', _ce),
                ('complex_category', FunctionTransformer(_complex_category_wrapper)),
                ('convert_category', FunctionTransformer(_convert_category)),
                ('drop_features', FunctionTransformer(_drop_features)),
                ('model', _model)
            ]

        return [
            ('count_encoding', _ce),
            ('complex_category', FunctionTransformer(_complex_category_wrapper)),
            ('drop_features', FunctionTransformer(_drop_features)),
            ('model', _model)
        ]

    if model_name in _clf_model:
        _bc_threshold = {
            'clf_low': 3.17,
            'clf_middle': 9.87,
            'clf_high': 10.70,
            'clf_very_high': 27.37
        }[model_name]

        _model = _ClassifierWrapper(
            lgb.LGBMClassifier(**model_params[model_name]),
            boxcox_threshold=get_transformed_threshold(_bc_threshold)
        )

        return [('model', _model)]

    match model_name:
        case 'kneighbors':
            _model = KNeighborsRegressor(**model_params['kneighbors'])
        case 'ridge':
            _model = Ridge(**model_params['ridge'])
        case 'lasso':
            _model = Lasso(**model_params['lasso'])
        case 'elasticnet':
            _model = ElasticNet(**model_params['elasticnet'])

    def _drop_features(X):
        return X.drop(columns=[
            'has_child', 'exp_per_hsize', 'any_nonagoric_and_sewer', 'lower_than_not_have_consumed',
            'hsize_diff_survey', 'hsize_ratio_survey', 'hsize_rank_survey', 'zscore_consumed_to_strata',
            'dependency_interaction', 'svd_complex_0', 'svd_complex_1', 'svd_complex_2'
        ])

    return [('drop_features', FunctionTransformer(_drop_features)), ('model', _model)]

# sub functions for preprocessing
def _drop_unused_columns(X):
    return X.drop(columns=['hhid', 'com', 'share_secondary'])


def _handle_null_numbers(X):
    _null_columns = ['utl_exp_ppp17']
    X = X.copy()
    for _col in _null_columns:
        X[_col] = X[_col].fillna(X[_col].mean())

    return X


def _complex_feature_wrapper(X):
    df_complex = complex_numbers_dataframe(X)

    df_complex.index = X.index
    return pd.concat([X, df_complex], axis=1)


def _complex_category_wrapper(X):
    df_complex_cat = preprocess.complex_category_dataframe(X)

    df_complex_cat.index = X.index
    return pd.concat([X, df_complex_cat], axis=1)


def _survey_related_feature_wrapper(X):
    df_survey_related = survey_related_features(X)

    df_survey_related.index = X.index
    return pd.concat([X, df_survey_related], axis=1)


def _drop_temporary_used_columns(X):
    return X.drop(columns=['survey_id'])


class _CustomCategoryMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict, columns):
        self.mapping_dict = mapping_dict
        self.columns = columns
        self.imputer = SimpleImputer(strategy='most_frequent')

    def fit(self, X, y=None):
        # 入力がDataFrameであることを確認
        X_cat = X[self.columns].copy()

        # マッピングを適用（未定義の値はNaNにする）
        for col in self.columns:
            X_cat[col] = X_cat[col].map(self.mapping_dict.get(col, {}))

        # 補完モデルを学習
        self.imputer.fit(X_cat)
        return self

    def transform(self, X):
        X_cat = X[self.columns].copy()

        # マッピングの適用
        for col in self.columns:
            X_cat[col] = X_cat[col].map(self.mapping_dict.get(col, {}))

        # 欠損値補完
        X_imputed = self.imputer.transform(X_cat)

        # 四捨五入して整数型に（元の関数のロジックを維持）
        return np.round(X_imputed).astype(int)

    def get_feature_names_out(self, input_features=None):
        return self.columns


class _SVDFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, consumed_svd=None, infra_svd=None, complex_svd=None):
        self.consumed_svd = consumed_svd
        self.infra_svd = infra_svd
        self.complex_svd = complex_svd

    def fit(self, X, y=None):
        # 学習時（Train）にSVDをfitさせる
        _, self.consumed_svd = preprocess.consumed_svd_dataframe(X, svd=self.consumed_svd)
        _, self.infra_svd = preprocess.infrastructure_svd_dataframe(X, svd=self.infra_svd)
        _, self.complex_svd = preprocess.complex_svd_dataframe(X, svd=self.complex_svd)
        return self

    def transform(self, X):
        X = X.copy()

        df_cons, _ = preprocess.consumed_svd_dataframe(X, svd=self.consumed_svd)
        df_infra, _ = preprocess.infrastructure_svd_dataframe(X, svd=self.infra_svd)
        df_complex, _ = preprocess.complex_svd_dataframe(X, svd=self.complex_svd)

        df_cons.index = X.index
        df_infra.index = X.index
        df_complex.index = X.index

        return pd.concat([X, df_cons, df_infra, df_complex], axis=1)


class _ClassifierWrapper(RegressorMixin, BaseEstimator):

    def __init__(self, classifier, boxcox_threshold=9.87):
        self.classifier = classifier
        self.boxcox_threshold = boxcox_threshold

    def fit(self, X, y):
        y_bin = (y < self.boxcox_threshold).astype(int)
        self.classifier.fit(X, y_bin)
        return self

    def predict(self, X):

        return self.classifier.predict_proba(X)[:, 1]
