__all__ = ['get_datas']

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