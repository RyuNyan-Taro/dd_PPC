"""Data preprocess scripts for modeling.

ref: https://qiita.com/DS27/items/aa3f6d0f03a8053e5810
"""

__all__ = []

import pandas as pd

from sklearn.preprocessing import StandardScaler


def standardized_with_numbers(train: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardized with number columns and return the values and StandardScaler model for prediction.

    Args:
        train: The training data

    Returns:
        Standardized and number columns only selected DataFrame and the StandardScaler model to use the prediction process.
    """

    _num_cols = ['hhid', 'com', 'weight', 'strata', 'hsize',
       'num_children5', 'num_children10', 'num_children18', 'age',
       'num_adult_female', 'num_adult_male', 'num_elderly',]

    x_train = train[_num_cols]

    sc = StandardScaler()
    sc.fit(x_train)

    x_train_std = sc.transform(x_train)

    return x_train_std, sc
