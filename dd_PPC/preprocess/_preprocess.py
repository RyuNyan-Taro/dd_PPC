"""Data preprocess scripts for modeling.

ref: https://qiita.com/DS27/items/aa3f6d0f03a8053e5810
"""

__all__ = ['standardized_with_numbers', 'standardized_with_numbers_dataframe','encoding_category', 'encoding_category_dataframe']

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def standardized_with_numbers_dataframe(train: pd.DataFrame, fit_model: StandardScaler | None = None) -> tuple[pd.DataFrame, StandardScaler]:
    x_train_std, _num_cols = _standardized(train, fit_model)

    return pd.DataFrame(x_train_std, columns=_num_cols), fit_model


def standardized_with_numbers(train: pd.DataFrame, fit_model: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    """Standardized with number columns and return the values and StandardScaler model for prediction.

    Args:
        train: The training data
        fit_model: The StandardScaler model to fit and transform the data. If None, a new model is created.

    Returns:
        Standardized and number columns only selected DataFrame and the StandardScaler model to use the prediction process.
    """

    x_train_std, _ = _standardized(train, fit_model)

    return x_train_std, fit_model


def _standardized(train: pd.DataFrame, fit_model: StandardScaler | None = None) -> tuple[np.ndarray, list[str]]:

    num_cols = ['weight', 'strata', 'hsize', 'age',
                 'num_children5', 'num_children10', 'num_children18',
                 'num_adult_female', 'num_adult_male', 'num_elderly', 'sworkershh', 'sfworkershh']

    x_train = train[num_cols]

    print('\nstandardize_numbers\n-----------------\n', x_train.head())

    if fit_model is None:
        fit_model = StandardScaler()
        fit_model.fit(x_train)

    x_train_std = fit_model.transform(x_train)

    return x_train_std, num_cols


def encoding_category(train: pd.DataFrame) -> np.ndarray:
    """
    Encodes categorical columns in the input DataFrame by converting specific category values into
    binary indicators.

    Args:
        train: The training data.

    Returns:
        pd.DataFrame: New DataFrame containing binary-encoded values for the specified
            categorical columns.
    """

    x_train, _ = _category_encoding(train)

    return x_train


def encoding_category_dataframe(train: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns in the input DataFrame by converting specific category values into
    binary indicators.

    Args:
        train: The training data.

    Returns:
        pd.DataFrame: New DataFrame containing binary-encoded values for the specified
            categorical columns.
    """

    x_train, _columns = _category_encoding(train)

    return pd.DataFrame(x_train, columns=_columns)


def _category_encoding(train: pd.DataFrame) -> tuple[np.ndarray, list[str]]:

    _access_or_not = {'Access': 1, 'No access': 0}
    _already_number = {0: 0, 1: 1}

    _category_number_maps = {
        'water': _access_or_not,
        'toilet': _access_or_not,
        'sewer': _access_or_not,
        'elect': _access_or_not,
        'male': {'Male': 1, 'Female': 0},
        'urban': {'Urban': 1, 'Rural': 0},
        'region1': _already_number,
        'region2': _already_number,
        'region3': _already_number,
        'region4': _already_number,
        'region5': _already_number,
        'region6': _already_number,
        'region7': _already_number,
    }

    category_cols = [
        'water', 'toilet', 'sewer', 'elect', 'male', 'urban',
        'region1', 'region2', 'region3', 'region4', 'region5', 'region6', 'region7',
    ]

    x_train = train[category_cols].copy()

    print('\nencoding_category\n-----------------\n', x_train.head())
    for _col in category_cols:
        x_train[_col] = x_train[_col].apply(lambda x: _category_number_maps[_col][x]).astype(int)

    return x_train.to_numpy(), category_cols


def create_new_features_array(df: pd.DataFrame) -> np.ndarray:

    _regions, _ = _create_features(df)

    return _regions


def create_new_features_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    _regions, _columns = _create_features(df)

    return pd.DataFrame(_regions, columns=_columns)


def _create_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:

    _regions = np.array([0] * len(df))
    for _col in df.columns:
        if _col.startswith('region_'):
            continue
        _region_number = int(_col.split('_')[1])
        _regions[_regions == 1] = _region_number

    return _regions, ['region']