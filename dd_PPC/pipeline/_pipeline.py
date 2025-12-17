__all__ = ['apply_random_forest', 'fit_and_predict_random_forest', 'pred_random_forest']

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .. import file, preprocess, model, data, calc

def apply_random_forest() -> tuple[RandomForestRegressor, np.ndarray, StandardScaler]:
    _datas = file.get_datas()

    _datas_std, sc = preprocess.standardized_with_numbers(_datas['train'])
    _datas_category = preprocess.encoding_category(_datas['train'])

    _x_train = np.hstack([_datas_std, _datas_category])
    _y_train = _datas['target_consumption'].loc[:, 'cons_ppp17']

    RF, pred_RF = model.fit_random_forest(_x_train, _y_train)

    return RF, pred_RF, sc


def fit_and_predict_random_forest():

    def fit_data(train_x_, train_cons_y_):
        _datas_std, sc = preprocess.standardized_with_numbers(train_x_)
        _datas_category = preprocess.encoding_category(train_x_)

        _x_train = np.hstack([_datas_std, _datas_category])
        _y_train = train_cons_y_['target_consumption'].loc[:, 'cons_ppp17']

        RF, pred_RF = model.fit_random_forest(_x_train, _y_train)

        return RF, pred_RF, sc

    def pred_data(test_x_, test_cons_y_, sc, rf):
        _datas_std, _ = preprocess.standardized_with_numbers(test_x_, sc)
        _datas_category = preprocess.encoding_category(test_x_)

        x_test = np.hstack([_datas_std, _datas_category])

        pred_cons_y = rf.predict(x_test)

        y_test = test_cons_y_['target_consumption'].loc[:, 'cons_ppp17']

        consumption = test_x.copy()
        consumption['cons_pred'] = pred_cons_y

        pred_rate_y = calc.poverty_rates_from_consumption(consumption, 'cons_pred')

        return x_test, y_test, consumption, pred_cons_y, pred_rate_y

    def show_metrics(pred_cons_y, y_test, pred_rate_y, consumption, rf, x_test, test_rate_y_):
        print(f'RMSE: {np.sqrt(np.mean((pred_cons_y - y_test) ** 2))}')
        print(f'MAE: {np.mean(np.abs(pred_cons_y - y_test))}')
        print(f'R2: {rf.score(x_test, y_test)}')
        print(
            f'CompetitionScore: {calc.weighted_average_of_consumption_and_poverty_rate(consumption, pred_rate_y, test_rate_y_)}')

    _datas = file.get_datas()

    train_x, train_cons_y, _, test_x, test_cons_y, test_rate_y = data.split_datas(_datas['train'],
                                                                                  _datas['target_consumption'],
                                                                                  _datas['target_rate'])

    _RF, _, _sc = fit_data(train_x, train_cons_y)

    _x_test, _y_test, _consumption, _pred_cons_y, _pred_rate_y = pred_data(test_x, test_cons_y, _sc, _RF)

    show_metrics(_pred_cons_y, _y_test, _pred_rate_y, _consumption, _RF, _x_test, test_rate_y)


def pred_random_forest(fit_model: RandomForestRegressor, sc: StandardScaler):
    _datas = file.get_datas()

    _datas_std, _ = preprocess.standardized_with_numbers(_datas['test'], sc)
    _datas_category = preprocess.encoding_category(_datas['test'])

    return fit_model.predict(np.hstack([_datas_std, _datas_category]))