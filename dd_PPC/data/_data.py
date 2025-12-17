__all__ = ['divide_value_types', 'split_datas']

import pandas as pd


def divide_value_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Divide the value types of the columns into two categories and return them."""

    two_categories = []
    some_category_or_number = []

    for _col in df.columns:
        _has_null = df[_col].isnull().sum() > 0
        _unique_len = df[_col].unique().shape[0]

        _unique_len_threshold = 3 if _has_null else 2
        _add_category = two_categories if _unique_len <= _unique_len_threshold else some_category_or_number
        _add_category.append(_col)

    print(len(two_categories), len(some_category_or_number))

    return two_categories, some_category_or_number


def split_datas(dependent: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _test_survey_id = 300000

    return (dependent[dependent.survey_id != _test_survey_id, :],
            target[target.survey_id != _test_survey_id],
            dependent[dependent.survey_id == _test_survey_id],
            target[target.survey_id == _test_survey_id])
