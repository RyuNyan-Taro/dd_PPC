__all__ = ['apply_random_forest']

import numpy as np
import pandas as pd

from .. import file, preprocess, model

def apply_random_forest():
    _datas = file.get_datas()

    _datas_std, _ = preprocess.standardized_with_numbers(_datas['train'])
    _datas_category = preprocess.encoding_category(_datas['train'])

    _x_train = np.hstack([_datas_std, _datas_category])
    _y_train = _datas['target_consumption'].loc[:, 'cons_ppp17']

    RF, pred_RF = model.fit_random_forest(_x_train, _y_train)

    return RF, pred_RF