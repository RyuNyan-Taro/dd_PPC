__all__ = ['apply_random_forest']

from .. import file, preprocess, model

def apply_random_forest():
    _datas = file.get_datas()
    _datas_std = preprocess.standardized_with_numbers(_datas['train'])
    RF, pred_RF = model.fit_random_forest(_datas_std, _datas['target_consumption'])

    return RF, pred_RF