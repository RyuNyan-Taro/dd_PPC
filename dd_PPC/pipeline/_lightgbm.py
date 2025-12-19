__all__ = ['apply_lightgbm']

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from .. import file, preprocess, model

def apply_lightgbm(show_pred_plot: bool = False) -> tuple[lgb.LGBMRegressor, np.ndarray, StandardScaler]:
    _datas = file.get_datas()

    _datas_std, sc = preprocess.standardized_with_numbers_dataframe(_datas['train'])
    _datas_category = preprocess.encoding_category_dataframe(_datas['train'])

    _x_train = np.hstack([_datas_std, _datas_category])
    _y_train = _datas['target_consumption'].loc[:, 'cons_ppp17'].to_numpy()

    LB, pred_LB = model.fit_lightgbm(_x_train, _y_train, show_pred_plot=show_pred_plot)

    return LB, pred_LB, sc