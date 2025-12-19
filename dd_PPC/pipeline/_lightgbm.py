__all__ = ['apply_lightgbm']

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from .. import file, preprocess, model

def apply_lightgbm() -> tuple[lgb.LGBMRegressor, np.ndarray, StandardScaler]:
    _datas = file.get_datas()

    _datas_std, sc = preprocess.standardized_with_numbers(_datas['train'])
    _datas_category = preprocess.encoding_category(_datas['train'])

    _x_train = np.hstack([_datas_std, _datas_category])
    _y_train = _datas['target_consumption'].loc[:, 'cons_ppp17']

    LB, pred_LB = model.fit_lightgbm(_x_train, _y_train)

    return LB, pred_LB, sc