__all__ = ['fit_and_test_pipeline']


import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, QuantileRegressor, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn import set_config
import lightgbm as lgb
import xgboost as xgb
import catboost

from .. import file, model, data, calc


def fit_and_test_pipeline():

    def get_bc_threshold(original_val, lam):
        return calc.apply_boxcox_transform(np.array([original_val]), lam)[0][0]

    def calculate_metrics(pred_cons_y, y, pred_rate_y, consumption, model_pipelines: list, X, target_rate_y) -> dict[str, float]:
        rmse = np.sqrt(np.mean((pred_cons_y - y) ** 2))
        mae = np.mean(np.abs(pred_cons_y - y))
        competition_score = calc.weighted_average_of_consumption_and_poverty_rate(consumption, pred_rate_y, target_rate_y)

        return dict(
            rmse=rmse,
            mae=mae,
            competition_score=competition_score
        )

    def plot_model_bias(y_true, y_pred, model_name):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.3, s=10)
        plt.plot([0, 50], [0, 50], '--', color='red')  # 理想線

        # 指標で使われている重要な閾値を描画
        _poverty_thresholds = [
            3.17, 9.13, 9.87, 10.70, 27.37
        ]
        for t in _poverty_thresholds:
            plt.axvline(t, color='green', linestyle=':', alpha=0.5)
            plt.axhline(t, color='green', linestyle=':', alpha=0.5)

        plt.xlabel('True Consumption')
        plt.ylabel('Predicted Consumption')
        plt.title(f'Bias Analysis: {model_name}')
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.show()

    boxcox_lambda = 0.09

    num_cols, category_cols, category_number_maps = _get_columns()
    model_params = _get_model_params()

    for _model, _params in model_params.items():
        print('model:', _model)
        print('params:', _params)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),

            ('cat', CustomCategoryMapper(category_number_maps, category_cols), category_cols)
        ],
        remainder='drop'
    )

    model_pipelines = [
        (
            'lgb', Pipeline([('prep', preprocessor), ('model', lgb.LGBMRegressor(**model_params['lightgbm']))])
        ),
        (
            'xgb', Pipeline([('prep', preprocessor), ('model', xgb.XGBRegressor(**model_params['xgboost']))])
        ),
        (
            'catboost',
            Pipeline([('prep', preprocessor), ('model', catboost.CatBoostRegressor(**model_params['catboost']))])
        ),
        (
            'ridge',
            Pipeline([('prep', preprocessor), ('model', Ridge(**model_params['ridge']))])
        ),
        # (
        #     'knn',
        #     Pipeline([('prep', preprocessor), ('model', KNeighborsRegressor(**model_params['kneighbors']))])
        # ),
        (
            'lasso',
            Pipeline([('prep', preprocessor), ('model',  Lasso(**model_params['lasso']))])
        ),
        # (
        #     'tabular',
        #     Pipeline([
        #         ('prep', preprocessor),
        #         ('convert', model.Float32Transformer()),
        #         ('model', model.get_tabular_nn_regressor(model_params['tabular']))
        #     ])
        # )
        (
            'mlp',
            Pipeline([
                ('prep', preprocessor),
                ('convert', model.Float32Transformer()),
                ('model', model.get_mlp_nn_regressor(model_params['mlp']))
            ])
        )
        # (
        #     'clf_low',
        #     Pipeline([('prep', preprocessor), (
        #         'model', ClassifierWrapper(lgb.LGBMClassifier(random_state=123, verbose = -1, force_row_wise=True),
        #                                    boxcox_threshold=get_bc_threshold(3.17, boxcox_lambda))
        #     )])
        # ),
        # (
        #     'clf_middle',
        #     Pipeline([('prep', preprocessor), (
        #         'model', ClassifierWrapper(lgb.LGBMClassifier(random_state=123, verbose=-1, force_row_wise=True),
        #                                    boxcox_threshold=get_bc_threshold(9.87, boxcox_lambda))
        #     )])
        # ),
        # (
        #     'clf_high',
        #     Pipeline([('prep', preprocessor), (
        #         'model', ClassifierWrapper(lgb.LGBMClassifier(random_state=123, verbose=-1, force_row_wise=True),
        #                                    boxcox_threshold=get_bc_threshold(10.70, boxcox_lambda))
        #     )])
        # ),
        # (
        #     'clf_very_high',
        #     Pipeline([('prep', preprocessor), (
        #         'model', ClassifierWrapper(lgb.LGBMClassifier(random_state=123, verbose=-1, force_row_wise=True),
        #                                    boxcox_threshold=get_bc_threshold(27.37, boxcox_lambda))
        #     )])
        # ),
        # (
        #     'elasticnet',
        #     Pipeline([('prep', preprocessor), ('model', ElasticNet(**model_params['elasticnet']))]),
        # )
    ]

    stacking_regressor = StackingRegressor(
        estimators=model_pipelines,
        final_estimator=Ridge(random_state=123, max_iter=10000),
        # final_estimator=HuberRegressor(max_iter=10000, epsilon=1.1),
        # final_estimator=Lasso(**model_params['lasso']),
        # final_estimator=QuantileRegressor(quantile=0.5),
        n_jobs=3,
        verbose=1
    )

    _datas = file.get_datas()

    _k_fold_test_ids = [100000, 200000, 300000]

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
        weights = stacking_regressor.final_estimator_.coef_
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

        _train_metrics = calculate_metrics(_y_train_mean_pred, train_cons_y.cons_ppp17, train_pred_rate_y, consumption, model_pipelines, train_x, train_rate_y)

        consumption = test_cons_y.copy()
        consumption['cons_pred'] = _y_test_mean_pred
        test_pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        test_pred_rate_y = model.transform_isotonic_regression(test_pred_rate_y, ir)

        _test_metrics = calculate_metrics(_y_test_mean_pred, test_cons_y.cons_ppp17, test_pred_rate_y, consumption, model_pipelines, test_x, test_rate_y)

        print(_train_metrics)
        print(_test_metrics)

        plot_model_bias(_y_test_mean_pred, test_cons_y.cons_ppp17, "Stacking Regressor")


class CustomCategoryMapper(BaseEstimator, TransformerMixin):
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


class ClassifierWrapper(RegressorMixin, BaseEstimator):

    def __init__(self, classifier, boxcox_threshold=9.87):
        self.classifier = classifier
        self.boxcox_threshold = boxcox_threshold

    def fit(self, X, y):
        y_bin = (y < self.boxcox_threshold).astype(int)
        self.classifier.fit(X, y_bin)
        return self

    def predict(self, X):

        return self.classifier.predict_proba(X)[:, 1]


def _get_columns() -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    _access_or_not = {'Access': 1, 'No access': 0}
    _already_number = {0: 0, 1: 1}
    _yes_no = {'Yes': 1, 'No': 0}

    category_number_maps = {
        'owner': {'Owner': 1, 'Not owner': 0},
        'water': _access_or_not,
        'toilet': _access_or_not,
        'sewer': _access_or_not,
        'elect': _access_or_not,
        'male': {'Male': 1, 'Female': 0},
        'urban': {'Urban': 1, 'Rural': 0},
        'employed': {'Employed': 1, 'Not employed': 0},
        'any_nonagric': _yes_no,
        'region1': _already_number,
        'region2': _already_number,
        'region3': _already_number,
        'region4': _already_number,
        'region5': _already_number,
        'region6': _already_number,
        'region7': _already_number,
        'water_source': {
            'Piped water into dwelling': 6,
            'Surface water': 1,
            'Other': 2,
            'Piped water to yard/plot': 5,
            'Protected dug well': 4,
            'Public tap or standpipe': 3,
            'Tanker-truck': 0,
            'Protected spring': 4,
        },
        'sanitation_source': {
            'A piped sewer system': 5,
            'A septic tank': 3,
            'Pit latrine with slab': 2,
            'No facilities or bush or field': 0,
            'Pit latrine': 1,
            'Other': 4,
        },
        'dweltyp': {
            'Detached house': 4,
            'Separate apartment': 2,
            'Several buildings connected': 1,
            'Other': 0,
            'Improvised housing unit': 3,
        },
        'educ_max': {
            'Complete Tertiary Education': 6,
            'Complete Secondary Education': 5,
            'Incomplete Tertiary Education': 4,
            'Incomplete Primary Education': 2,
            'Complete Primary Education': 3,
            'Incomplete Secondary Education': 1,
            'Never attended': 0,
        },
        'sector1d': {
            'Agriculture, hunting and forestry': 0,
            'Wholesale and retail trade': 1,
            'Transport, storage and communications': 2,
            'Manufacturing': 3,
            'Construction': 4,
            'Public administration and defence': 5,
            'Hotels and restaurants': 6,
            'Education': 7,
            'Real estate, renting and business activities': 8,
            'Other community, social and personal service activities': 9,
            'Mining and quarrying': 10,
            'Health and social work': 11,
            'Activities of private households as employers ': 12,
            'Fishing': 13,
            'Financial intermediation': 14,
            'Electricity, gas and water supply': 15,
            ' Extraterritorial organizations and bodies': 16,
        },
        'consumed100': _yes_no, 'consumed200': _yes_no, 'consumed300': _yes_no, 'consumed400': _yes_no,
        'consumed500': _yes_no, 'consumed600': _yes_no, 'consumed700': _yes_no, 'consumed800': _yes_no,
        'consumed900': _yes_no, 'consumed1000': _yes_no, 'consumed1100': _yes_no, 'consumed1200': _yes_no,
        'consumed1300': _yes_no, 'consumed1400': _yes_no, 'consumed1500': _yes_no, 'consumed1600': _yes_no,
        'consumed1700': _yes_no, 'consumed1800': _yes_no, 'consumed1900': _yes_no, 'consumed2000': _yes_no,
        'consumed2100': _yes_no, 'consumed2200': _yes_no, 'consumed2300': _yes_no, 'consumed2400': _yes_no,
        'consumed2500': _yes_no, 'consumed2600': _yes_no, 'consumed2700': _yes_no, 'consumed2800': _yes_no,
        'consumed2900': _yes_no, 'consumed3000': _yes_no, 'consumed3100': _yes_no, 'consumed3200': _yes_no,
        'consumed3300': _yes_no, 'consumed3400': _yes_no, 'consumed3500': _yes_no, 'consumed3600': _yes_no,
        'consumed3700': _yes_no, 'consumed3800': _yes_no, 'consumed3900': _yes_no, 'consumed4000': _yes_no,
        'consumed4100': _yes_no, 'consumed4200': _yes_no, 'consumed4300': _yes_no, 'consumed4400': _yes_no,
        'consumed4500': _yes_no, 'consumed4600': _yes_no, 'consumed4700': _yes_no, 'consumed4800': _yes_no,
        'consumed4900': _yes_no, 'consumed5000': _yes_no,

    }

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

    num_cols = ['weight', 'strata', 'hsize', 'age',
                'num_children5', 'num_children10', 'num_children18',
                'num_adult_female', 'num_adult_male', 'num_elderly', 'sworkershh', 'sfworkershh']

    return num_cols, category_cols, category_number_maps


def _get_model_params() -> dict[str, dict]:
    model_params = {}
    for _model in ['lightgbm', 'xgboost', 'catboost', 'ridge', 'lasso', 'elasticnet']:
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

    return model_params
