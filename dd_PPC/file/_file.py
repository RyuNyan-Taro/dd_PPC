__all__ = ['get_datas', 'save_to_submission_format', 'get_submission_formats', 'load_best_params']

import os
import joblib
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


def save_to_submission_format(predictions: np.ndarray, pred_rate: pd.DataFrame | None = None, folder_prefix: str | None = None):
    """Saves predictions to specified file paths.
    Args:
        predictions: Predictions returned by a model.
        pred_rate: Poverty rates calculated from predictions.
        folder_prefix: If selected, it is added as the prefix of the save folder.

    """
    if folder_prefix is None:
        folder_prefix = ''

    _dir_path = '../results/'

    _consumption_format, _poverty_distribution_format = get_submission_formats(_dir_path)

    _consumption_format['cons_ppp17'] = predictions

    if pred_rate is None:
        _add_distribution(_poverty_distribution_format, predictions)
    else:
        pred_rate.columns = _poverty_distribution_format.columns
        _poverty_distribution_format = pred_rate

    _save_submissions(_consumption_format, _poverty_distribution_format, _dir_path, folder_prefix)


def get_submission_formats(dir_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    consumption_format = pd.read_csv(
        dir_path + '/Poverty_Prediction_Challenge_-_Submission_format/predicted_household_consumption.csv')
    poverty_distribution_format = pd.read_csv(
        dir_path + '/Poverty_Prediction_Challenge_-_Submission_format/predicted_poverty_distribution.csv')

    return consumption_format, poverty_distribution_format


def load_best_params(model_name: str, params_file: str = None) -> dict:
    """Load best parameters from a saved pickle file.

    Args:
        model_name: Name of the model ('xgboost', 'lightgbm', or 'catboost')
        params_file: Path to the pickle file. If None, uses default path.

    Returns:
        Dictionary of model parameters

    Example:
        >>> params = load_best_params('xgboost')
        >>> # Use with fit functions
        >>> model, pred = fit_xgboost(x_train, y_train, params=params)
    """

    if params_file is None:
        params_file = f'../models/optuna_study_{model_name}.pkl'

    if not os.path.exists(params_file):
        raise FileNotFoundError(
            f"Parameter file not found: {params_file}\n"
            f"Run tuning_model('{model_name}') first to generate the parameters."
        )

    try:
        params = joblib.load(params_file)

        return params.best_params

    except Exception as e:
        raise RuntimeError(f"Error loading parameters: {e}")


# internals for save_to_submission_format
def _add_distribution(poverty_distribution_format: pd.DataFrame, predictions: np.ndarray):
    _test = pd.read_csv('../datas/Poverty_Prediction_Challenge_-_Test_Data_-_Household_survey_features.csv')

    survey_400000_cond = _test.survey_id.to_numpy() == 400000
    survey_500000_cond = _test.survey_id.to_numpy() == 500000
    survey_600000_cond = _test.survey_id.to_numpy() == 600000

    for _col in poverty_distribution_format.columns[1:]:

        _percent = float(_col.split('_')[-1])
        poverty_distribution_format[_col] = [
            sum(predictions[_cond] <= _percent) / sum(_cond)
            for _cond in
            [survey_400000_cond, survey_500000_cond, survey_600000_cond]
        ]


def _save_submissions(consumption_format: pd.DataFrame, poverty_distribution_format: pd.DataFrame, dir_path: str, folder_prefix: str):
    _folder_name = f'{folder_prefix}_{datetime.datetime.now().strftime("%y%m%d%H%M%S")}'
    _save_dir = dir_path + _folder_name
    os.mkdir(_save_dir)

    consumption_format.to_csv(os.path.join(_save_dir, 'predicted_household_consumption.csv'), index=False)
    poverty_distribution_format.to_csv(os.path.join(_save_dir, 'predicted_poverty_distribution.csv'), index=False)
