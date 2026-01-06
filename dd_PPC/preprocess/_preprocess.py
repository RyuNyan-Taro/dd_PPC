"""Data preprocess scripts for modeling.

ref: https://qiita.com/DS27/items/aa3f6d0f03a8053e5810
"""

__all__ = ['standardized_with_numbers', 'standardized_with_numbers_dataframe','encoding_category',
           'encoding_category_dataframe', 'create_new_features_data_frame', 'create_new_features_array']

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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
    _yes_no = {'Yes': 1, 'No': 0}

    _category_number_maps = {
        'owner': {'Owner': 1, 'Not owner': 0},
        'water': _access_or_not,
        'toilet': _access_or_not,
        'sewer': _access_or_not,
        'elect': _access_or_not,
        'male': {'Male': 1, 'Female': 0},
        'urban': {'Urban': 1, 'Rural': 0},
        'employed': {'Employed': 1, 'Not employed': 0},
        'any_nonagric': _yes_no,
        'region1': _already_number,
        'region2': _already_number,
        'region3': _already_number,
        'region4': _already_number,
        'region5': _already_number,
        'region6': _already_number,
        'region7': _already_number,
        'water_source': {
            'Piped water into dwelling': 6,
            'Surface water': 1,
            'Other': 2,
            'Piped water to yard/plot': 5,
            'Protected dug well': 4,
            'Public tap or standpipe': 3,
            'Tanker-truck': 0,
            'Protected spring': 4,
        },
        'sanitation_source': {
            'A piped sewer system': 5,
            'A septic tank': 3,
            'Pit latrine with slab': 2,
            'No facilities or bush or field': 0,
            'Pit latrine': 1,
            'Other': 4,
        },
        'dweltyp': {
            'Detached house': 4,
            'Separate apartment': 2,
            'Several buildings connected': 1,
            'Other': 0,
            'Improvised housing unit': 3,
        },
        'educ_max': {
            'Complete Tertiary Education': 6,
            'Complete Secondary Education': 5,
            'Incomplete Tertiary Education': 4,
            'Incomplete Primary Education': 2,
            'Complete Primary Education': 3,
            'Incomplete Secondary Education': 1,
            'Never attended': 0,
        },
        'sector1d': {
            'Agriculture, hunting and forestry': 0,
            'Wholesale and retail trade': 1,
            'Transport, storage and communications': 2,
            'Manufacturing': 3,
            'Construction': 4,
            'Public administration and defence': 5,
            'Hotels and restaurants': 6,
            'Education': 7,
            'Real estate, renting and business activities': 8,
            'Other community, social and personal service activities': 9,
            'Mining and quarrying': 10,
            'Health and social work': 11,
            'Activities of private households as employers ': 12,
            'Fishing': 13,
            'Financial intermediation': 14,
            'Electricity, gas and water supply': 15,
            ' Extraterritorial organizations and bodies': 16,
        },

    }

    category_cols = [
        'water', 'toilet', 'sewer', 'elect', 'water_source',
        'male', 'urban',
        'owner', 'employed', 'any_nonagric',
        'sanitation_source', 'dweltyp', 'educ_max', 'sector1d',
        'region1', 'region2', 'region3', 'region4', 'region5', 'region6', 'region7',
    ]

    x_train = train[category_cols].copy()

    for _col in category_cols:

        _nulls = x_train[_col].isnull()
        x_train.loc[~_nulls, _col] = x_train.loc[~_nulls, _col].apply(lambda x: _category_number_maps[_col][x])

    _knn_imputed = pd.DataFrame(
        KNNImputer(n_neighbors=2).fit_transform(train),
        columns=train.columns
    )
    _iterative_imputed = pd.DataFrame(
        IterativeImputer(max_iter=20, random_state=0).fit_transform(train),
        columns=train.columns
    )
    _imputed = _knn_imputed

    for _col in category_cols:

        x_train[_col] = _knn_imputed[_col].apply(round).astype(int)

    return x_train.to_numpy(), category_cols


def create_new_features_array(df: pd.DataFrame) -> np.ndarray:

    _features, _ = _create_infra_features(df)

    return _features


def create_new_features_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    _infra_features, _infra_columns = _create_infra_features(df)

    # _features, _columns = _create_interaction_features(df)

    _binned_features, _binned_columns = _create_binned_features(df)

    _datas = [
        pd.DataFrame(_features, columns=_columns).reset_index(drop=True)
        for _features, _columns
        in [(_infra_features, _infra_columns), (_binned_features, _binned_columns)]
    ]

    return pd.concat(_datas, axis=1)


def _create_infra_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:

    # it worse the score to 9.667269918408259 -> 9.871187466282631
    features = df.copy()

    _infra_columns = ['water', 'toilet', 'sewer', 'elect']

    features['infra_count'] = df[_infra_columns].apply(lambda x: sum([{'Access': 1, 'No access': 0}[_val] for _val in x]), axis=1)

    # Infrastructure access patterns
    water_access = (df['water'] == 'Access').astype(int)
    toilet_access = (df['toilet'] == 'Access').astype(int)
    sewer_access = (df['sewer'] == 'Access').astype(int)
    elect_access = (df['elect'] == 'Access').astype(int)

    # Infrastructure combinations
    features['water_toilet'] = water_access * toilet_access
    features['full_sanitation'] = water_access * toilet_access * sewer_access
    features['modern_amenities'] = elect_access * (df['water_source'].isin(['Piped water to yard/plot', 'Piped water into dwelling'])).astype(int)

    # Infrastructure quality weighted by household size
    features['infra_per_person'] = (water_access + toilet_access + sewer_access + elect_access) / df['hsize']

    feature_columns = [
        'infra_count',
        'water_toilet',
        'full_sanitation',
        'modern_amenities',
        'infra_per_person'
    ]

    return features[feature_columns], feature_columns


def _create_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create interaction features between key variables."""

    # it worse the score to 9.667269918408259 -> 10.056012578908941
    features = df.copy()

    # Household composition interactions
    features['adults_per_child'] = (features['num_adult_female'] + features['num_adult_male']) / (
                features['num_children18'] + 1)
    features['dependency_ratio'] = features['num_children18'] / (
                features['num_adult_female'] + features['num_adult_male'] + 1)

    # Economic interactions
    employed = (df['employed'] == 'Employed').astype(int)
    _top_value = features['educ_max'].value_counts().idxmax()
    features['educ_max'] = features['educ_max'].fillna(_top_value)
    educ_max = features['educ_max'].apply(lambda x: {
            'Complete Tertiary Education': 6,
            'Complete Secondary Education': 5,
            'Incomplete Tertiary Education': 4,
            'Incomplete Primary Education': 2,
            'Complete Primary Education': 3,
            'Incomplete Secondary Education': 1,
            'Never attended': 0,
        }[x]).astype(int)

    features['workers_per_household'] = features['sworkershh'] / features['hsize']
    features['education_employment'] = educ_max * employed

    features_columns = [
        'adults_per_child',
        'dependency_ratio',
        'workers_per_household',
        'education_employment'
    ]

    return features[features_columns], features_columns


def _create_binned_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create binned versions of continuous variables."""

    # a little worse with dropping the base column: 9.667269918408259 -> 9.77464917925407
    features = df.copy()

    # Age groups
    features['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 65, 100], labels=[0, 1, 2, 3])

    # Household size categories
    features['hsize_category'] = pd.cut(df['hsize'], bins=[0, 2, 4, 6, 20], labels=[0, 1, 2, 3])

    # Polynomial features for key variables
    features['age_squared'] = df['age'] ** 2
    features['hsize_squared'] = df['hsize'] ** 2

    feature_columns = ['age_group', 'hsize_category', 'age_squared', 'hsize_squared']

    return features[feature_columns], feature_columns
