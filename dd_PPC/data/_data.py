__all__ = ['divide_value_types', 'split_datas', 'compare_transformations']

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import boxcox, yeojohnson


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


def split_datas(
        dependent: pd.DataFrame, consumptions: pd.DataFrame, rates: pd.DataFrame,
        test_survey_ids: list[int] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if test_survey_ids is None:
        test_survey_ids = [300000]

    return (dependent[~dependent.survey_id.isin(test_survey_ids)],
            consumptions[~consumptions.survey_id.isin(test_survey_ids)],
            rates[~rates.survey_id.isin(test_survey_ids)],
            dependent[dependent.survey_id.isin(test_survey_ids)],
            consumptions[consumptions.survey_id.isin(test_survey_ids)],
            rates[rates.survey_id.isin(test_survey_ids)],
            )


def compare_transformations(y_data: np.ndarray):
    """Compare different transformations visually and statistically"""

    # Original
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # log1p (current)
    log_transformed = np.log1p(y_data)
    ax1.hist(log_transformed, bins=50, alpha=0.7)
    ax1.set_title(f'log1p\nSkewness: {stats.skew(log_transformed):.3f}')

    # Box-Cox
    boxcox_transformed, lambda_bc = boxcox(y_data)
    ax2.hist(boxcox_transformed, bins=50, alpha=0.7)
    ax2.set_title(f'Box-Cox (λ={lambda_bc:.3f})\nSkewness: {stats.skew(boxcox_transformed):.3f}')

    # Yeo-Johnson
    yj_transformed, lambda_yj = yeojohnson(y_data)
    ax3.hist(yj_transformed, bins=50, alpha=0.7)
    ax3.set_title(f'Yeo-Johnson (λ={lambda_yj:.3f})\nSkewness: {stats.skew(yj_transformed):.3f}')

    plt.tight_layout()
    plt.show()

    print(f"\nLambda values:")
    print(f"Box-Cox: {lambda_bc:.4f}")
    print(f"Yeo-Johnson: {lambda_yj:.4f}")
    print(f"\nNote: log1p is equivalent to Box-Cox with λ=0")
