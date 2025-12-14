"""Data preprocess scripts for modeling.

ref: https://qiita.com/DS27/items/aa3f6d0f03a8053e5810
"""

__all__ = ['standardized_with_numbers', 'encoding_category']

import pandas as pd

from sklearn.preprocessing import StandardScaler


def standardized_with_numbers(train: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardized with number columns and return the values and StandardScaler model for prediction.

    Args:
        train: The training data

    Returns:
        Standardized and number columns only selected DataFrame and the StandardScaler model to use the prediction process.
    """

    _num_cols = ['weight', 'strata', 'hsize', 'age',
       'num_children5', 'num_children10', 'num_children18',
       'num_adult_female', 'num_adult_male', 'num_elderly', 'sworkershh', 'sfworkershh']

    x_train = train[_num_cols]

    sc = StandardScaler()
    sc.fit(x_train)

    x_train_std = sc.transform(x_train)

    return x_train_std, sc


def encoding_category(train: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns in the input DataFrame by converting specific category values into
    binary indicators.

    Args:
        train: The training data.

    Returns:
        pd.DataFrame: New DataFrame containing binary-encoded values for the specified
            categorical columns.
    """

    _category_cols = ['water', 'toilet', 'sewer', 'elect']

    x_train = train[_category_cols].copy()

    for _col in _category_cols:
        x_train[_col] = x_train[_col].apply(lambda x: 1 if x == 'Access' else 0).astype(int)

    return x_train
