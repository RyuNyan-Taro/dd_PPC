__all__ = ['get_datas']

import os
import shutil
import datetime
import numpy as np
import pandas as pd


def get_datas() -> dict:
    """Retrieves and loads multiple datasets from specified file paths.

    Returns:
        dict: A dictionary containing the following keys and corresponding pandas
            DataFrames:
            - 'train': Training data features.
            - 'test': Testing data features.
            - 'target_consumption': Target labels for household consumption.
            - 'target_rate': Target labels for poverty rates.
    """

    dir_path = '../datas/'
    _train = pd.read_csv(dir_path + 'Poverty_Prediction_Challenge_-_Training_Data_-_Household_survey_features.csv')
    _test = pd.read_csv(dir_path + 'Poverty_Prediction_Challenge_-_Test_Data_-_Household_survey_features.csv')
    _target_consumption = pd.read_csv(dir_path + 'Poverty_Prediction_Challenge_-_Training_Data_-_Household_consumption_labels.csv')
    _target_rate = pd.read_csv(dir_path + 'Poverty_Prediction_Challenge_-_Training_Data_-_Poverty_rate_labels.csv')

    return {'train': _train, 'test': _test, 'target_consumption': _target_consumption, 'target_rate': _target_rate}


def save_submission(predictions: np.ndarray, folder_prefix: str | None = None):
    """Saves predictions to specified file paths.
    Args:
        predictions: Predictions returned by a model.
        folder_prefix: If selected, it is added as the prefix of the save folder.

    """
    if folder_prefix is None:
        folder_prefix = ''

    _dir_path = '../results/'

    _consumption_format = pd.read_csv(
        _dir_path + '/Poverty_Prediction_Challenge_-_Submission_format/predicted_household_consumption.csv')
    _poverty_distribution_format = pd.read_csv(
        _dir_path + '/Poverty_Prediction_Challenge_-_Submission_format/predicted_poverty_distribution.csv')

    _consumption_format['cons_ppp17'] = predictions

    _test = pd.read_csv('../datas/Poverty_Prediction_Challenge_-_Test_Data_-_Household_survey_features.csv')

    survey_400000_cond = _test.survey_id.to_numpy() == 400000
    survey_500000_cond = _test.survey_id.to_numpy() == 500000
    survey_600000_cond = _test.survey_id.to_numpy() == 600000

    for _col in _poverty_distribution_format.columns[1:]:

        _percent = float(_col.split('_')[-1])
        _poverty_distribution_format[_col] = [
            sum(predictions[_cond] <= _percent) / sum(_cond)
            for _cond in
            [survey_400000_cond, survey_500000_cond, survey_600000_cond]
        ]

    _folder_name = folder_prefix + datetime.datetime.now().strftime('%y%m%d%H%M%S')
    _save_dir = _dir_path + _folder_name
    os.mkdir(_save_dir)

    _consumption_format.to_csv(_save_dir + 'predicted_household_consumption.csv', index=False)
    _poverty_distribution_format.to_csv(_save_dir + 'predicted_poverty_distribution.csv', index=False)

    shutil.make_archive(_folder_name, 'zip', _save_dir)
