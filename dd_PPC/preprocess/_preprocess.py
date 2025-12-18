"""Data preprocess scripts for modeling.

ref: https://qiita.com/DS27/items/aa3f6d0f03a8053e5810
"""

__all__ = ['standardized_with_numbers', 'encoding_category']

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


def standardized_with_numbers(train: pd.DataFrame, fit_model: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    """Standardized with number columns and return the values and StandardScaler model for prediction.

    Args:
        train: The training data
        fit_model: The StandardScaler model to fit and transform the data. If None, a new model is created.

    Returns:
        Standardized and number columns only selected DataFrame and the StandardScaler model to use the prediction process.
    """

    _num_cols = ['weight', 'strata', 'hsize', 'age',
       'num_children5', 'num_children10', 'num_children18',
       'num_adult_female', 'num_adult_male', 'num_elderly', 'sworkershh', 'sfworkershh']

    x_train = train[_num_cols]

    print('\nstandardize_numbers\n-----------------\n', x_train.head())

    if fit_model is None:
        fit_model = StandardScaler()
        fit_model.fit(x_train)

    x_train_std = fit_model.transform(x_train)

    return x_train_std, fit_model


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

    _access_or_not = {'Access': 1, 'No access': 0}

    _category_number_maps = {
        'water': _access_or_not,
        'toilet': _access_or_not,
        'sewer': _access_or_not,
        'elect': _access_or_not,
        # 'male': {'Male': 1, 'Female': 0},
        'urban': {'Urban': 1, 'Rural': 0},
    }

    _category_cols = ['water', 'toilet', 'sewer', 'elect', 'urban']

    x_train = train[_category_cols].copy()

    print('\nencoding_category\n-----------------\n', x_train.head())
    for _col in _category_cols:
        x_train[_col] = x_train[_col].apply(lambda x: _category_number_maps[_col][x]).astype(int)

    return x_train.to_numpy()
