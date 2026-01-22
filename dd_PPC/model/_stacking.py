__all__ = ['get_stacking_regressor_and_pipelines']

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, QuantileRegressor, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.impute import SimpleImputer
from category_encoders import CountEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost

from .. import file, model, calc
from .._config import CATEGORY_NUMBER_MAPS, NUMBER_COLUMNS
from ..preprocess import consumed_svd_dataframe, infrastructure_svd_dataframe, complex_numbers_dataframe, \
    survey_related_features


def get_stacking_regressor_and_pipelines(
        model_names: list[str],
        boxcox_lambda: float,
        model_params: dict | None = None
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

    num_cols, category_cols, category_number_maps = _get_columns()

    if model_params is None:
        model_params = _get_model_params(model_names)

    for _model, _params in model_params.items():
        print('model:', _model)
        print('params:', _params)

    preprocessor = _get_common_preprocess(category_number_maps, category_cols, num_cols)

    model_pipelines = [
        (
            _name,
            Pipeline([('prep', preprocessor)] + _get_initialized_model(_name, model_params, boxcox_lambda=boxcox_lambda))
        ) for _name in model_names]

    stacking_regressor = StackingRegressor(
        estimators=model_pipelines,
        # final_estimator=Ridge(random_state=123, max_iter=10000, positive=True, alpha=1, fit_intercept=True),
        final_estimator=HuberRegressor(max_iter=10000, epsilon=1.1),
        # final_estimator=lgb.LGBMRegressor(),
        # final_estimator=Lasso(**model_params['lasso']),
        # final_estimator=QuantileRegressor(quantile=0.5),
        n_jobs=2,
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

    _complex_input_cols = num_cols + [
        'svd_consumed_0', 'svd_infrastructure_0', 'urban', 'sanitation_source', 'svd_consumed_1', 'educ_max', 'sector1d',
        'consumed3100', 'consumed1500', 'consumed2000', 'consumed3000', 'consumed1800'
    ]

    complex_output_cols = list(complex_numbers_dataframe(pd.DataFrame(
        [[0] * len(_complex_input_cols), [1] * len(_complex_input_cols)],
        columns=_complex_input_cols)).columns)

    svd_cols = [f'svd_consumed_{i}' for i in range(3)] + [f'svd_infrastructure_{i}' for i in range(3)]

    _survey_cols = list({'survey_id', 'sanitation_source'} | set(num_cols) | {'educ_max'} | set(complex_output_cols) | set(svd_cols))

    survey_related_output_cols = list(survey_related_features(pd.DataFrame(
        [[0] * len(_survey_cols), [1] * len(_survey_cols)],
        columns=_survey_cols)).columns)

    final_num_cols = num_cols + svd_cols + complex_output_cols + survey_related_output_cols

    return final_num_cols, category_cols, CATEGORY_NUMBER_MAPS


def _get_model_params(model_names: list[str]) -> dict[str, dict]:
    model_params = {}
    for _model in ['lightgbm', 'xgboost', 'catboost', 'ridge', 'lasso', 'elasticnet', 'kneighbors']:
        _model_param = file.load_best_params(_model)
        match _model:
            case 'lightgbm':
                _model_param['objective'] = 'regression'
                _model_param['metric'] = 'rmse'
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


def _get_initialized_model(model_name: str, model_params: dict, boxcox_lambda: float) -> list[tuple[str, BaseEstimator]]:
    _add_float_size_conversion = ['tabular', 'mlp']
    _add_count_encoding = ['lightgbm', 'catboost', 'xgboost']
    _clf_model = ['clf_low', 'clf_middle', 'clf_high', 'clf_very_high']
    _model = None

    def get_bc_threshold(original_val, lam):
        return calc.apply_boxcox_transform(np.array([original_val]), lam)[0][0]

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
        match model_name:
            case 'lightgbm':
                _model = lgb.LGBMRegressor(**model_params['lightgbm'])
            case 'catboost':
                _model = catboost.CatBoostRegressor(**model_params['catboost'])
            case 'xgboost':
                _model = xgb.XGBRegressor(**model_params['xgboost'])
            case _:
                raise ValueError(f'Invalid model name: {model_name}')

        _count_encoding_cols = ['sector1d']

        _ce = ColumnTransformer(
            transformers=[(
                'encoding', CountEncoder(cols=_count_encoding_cols, handle_unknown=0, normalize=True), _count_encoding_cols)],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        if model_name == 'catboost':
            def _drop_catboost_features(X):
                return X.drop(columns=['water', 'sewer', 'urban'])

            return [('count_encoding', _ce), ('drop_features', FunctionTransformer(_drop_catboost_features)), ('model', _model)]

        return [('count_encoding', _ce), ('model', _model)]

        # return [('model', _model)]

    if model_name in _clf_model:
        _bc_threshold = {
            'clf_low': 3.17,
            'clf_middle': 9.87,
            'clf_high': 10.70,
            'clf_very_high': 27.37
        }[model_name]

        _model = _ClassifierWrapper(
            lgb.LGBMClassifier(**model_params[model_name]),
            boxcox_threshold=get_bc_threshold(_bc_threshold, boxcox_lambda)
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

    return [('model', _model)]

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
    def __init__(self, consumed_svd=None, infra_svd=None):
        self.consumed_svd = consumed_svd
        self.infra_svd = infra_svd

    def fit(self, X, y=None):
        # 学習時（Train）にSVDをfitさせる
        _, self.consumed_svd = consumed_svd_dataframe(X, svd=self.consumed_svd)
        _, self.infra_svd = infrastructure_svd_dataframe(X, svd=self.infra_svd)
        return self

    def transform(self, X):
        X = X.copy()

        df_cons, _ = consumed_svd_dataframe(X, svd=self.consumed_svd)
        df_infra, _ = infrastructure_svd_dataframe(X, svd=self.infra_svd)

        df_cons.index = X.index
        df_infra.index = X.index

        return pd.concat([X, df_cons, df_infra], axis=1)


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
