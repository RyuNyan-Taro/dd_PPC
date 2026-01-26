"""Data preprocess scripts for modeling.

ref: https://qiita.com/DS27/items/aa3f6d0f03a8053e5810
"""

__all__ = ['standardized_with_numbers', 'standardized_with_numbers_dataframe','encoding_category',
           'encoding_category_dataframe', 'create_new_features_data_frame', 'create_new_features_array',
           'target_encode', 'create_survey_aggregates', 'consumed_svd_dataframe', 'infrastructure_svd_dataframe', 'complex_numbers_dataframe', 'survey_related_features']

import numpy as np
import pandas as pd
import tqdm
from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def standardized_with_numbers_dataframe(train: pd.DataFrame, fit_model: StandardScaler | None = None, add_columns: list[str] | None = None) -> tuple[pd.DataFrame, StandardScaler]:
    x_train_std, _num_cols = _standardized(train, fit_model, add_columns)

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


def _standardized(train: pd.DataFrame, fit_model: StandardScaler | None = None, add_columns: list[str] | None = None) -> tuple[np.ndarray, list[str]]:

    num_cols = ['weight', 'strata', 'hsize', 'age', 'utl_exp_ppp17',
                 'num_children5', 'num_children10', 'num_children18',
                 'num_adult_female', 'num_adult_male', 'num_elderly', 'sworkershh', 'sfworkershh']

    if add_columns is not None:
        num_cols = num_cols + add_columns

    x_train = train[num_cols].copy()

    for _col in num_cols:
        if x_train[_col].isnull().sum() > 0:
            x_train[_col] = x_train[_col].fillna(x_train[_col].mean())

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


def consumed_svd_dataframe(train: pd.DataFrame, n_components: int = 3, svd: TruncatedSVD | None = None) -> tuple[pd.DataFrame, TruncatedSVD]:
    latent_feats, svd, columns = _consumed_svd(train, n_components, svd)

    if isinstance(latent_feats, pd.DataFrame):
        return pd.DataFrame(latent_feats.to_numpy(), columns=columns), svd

    return pd.DataFrame(latent_feats, columns=columns), svd

def _consumed_svd(train: pd.DataFrame, n_components, svd: TruncatedSVD | None = None) -> tuple[pd.DataFrame, TruncatedSVD, list[str]]:
    consumed_cols = [c for c in train.columns if 'consumed' in c]

    if svd is None:
        svd = TruncatedSVD(n_components=n_components, random_state=123)
        svd.fit(train[consumed_cols])

    latent_feats = svd.transform(train[consumed_cols])

    columns = [f'svd_consumed_{_i}' for _i in range(n_components)]

    return latent_feats, svd, columns


def infrastructure_svd_dataframe(train: pd.DataFrame, n_components: int = 3, svd: TruncatedSVD | None = None) -> tuple[pd.DataFrame, TruncatedSVD]:
    latent_feats, svd, columns = _infrastructure_svd(train, n_components, svd)

    if isinstance(latent_feats, pd.DataFrame):
        return pd.DataFrame(latent_feats.to_numpy(), columns=columns), svd

    return pd.DataFrame(latent_feats, columns=columns), svd

def _infrastructure_svd(train: pd.DataFrame, n_components, svd: TruncatedSVD | None = None) -> tuple[pd.DataFrame, TruncatedSVD, list[str]]:
    infrastructure_cols = [
        'water', 'toilet', 'sewer', 'elect', 'water_source', 'sector1d'
    ]

    if svd is None:
        svd = TruncatedSVD(n_components=n_components, random_state=123)
        svd.fit(train[infrastructure_cols])

    latent_feats = svd.transform(train[infrastructure_cols])

    columns = [f'svd_infrastructure_{_i}' for _i in range(n_components)]

    return latent_feats, svd, columns


def complex_numbers_dataframe(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    _strata_mean = train.groupby('strata')['svd_consumed_0'].transform('mean')
    _strata_std = train.groupby('strata')['svd_consumed_0'].transform('std')
    # _infra_strata_mean = train.groupby('strata')['svd_infrastructure_0'].transform('mean')
    _adult_equivalence = 1 + 0.7 * (train['num_adult_male'] + train['num_adult_female'] - 1) + 0.5 * (train['num_children5'] + train['num_children10'] + train['num_children18'])
    _sector_edu_mean = train.groupby('sector1d')['educ_max'].transform('mean')
    _strata_edu_mean = train.groupby('strata')['educ_max'].transform('mean')

    train['has_child'] = (train['num_children5'] + train['num_children10'] + train['num_children18'] > 0).apply(lambda x: 1 if x else 0)
    _consumed_cols = [c for c in train.columns if 'consumed' in c and c.startswith('consumed')]

    # _strata_infra_mean = train.groupby('strata')['svd_infrastructure_0'].transform('mean')
    # train['relative_infra_wealth'] = train['svd_infrastructure_0'] - _strata_infra_mean

    # _sector_mean = train.groupby('sector1d')['svd_consumed_0'].transform('mean')
    # _sector_std = train.groupby('sector1d')['svd_consumed_0'].transform('std')

    _complex_numbers = {
        # 'adult_equivalence': _adult_equivalence,
        'strata_times_infra': train['strata'] * train['svd_infrastructure_0'],
        'sanitation_and_consumed': (train['sanitation_source'] + 1) * train['svd_consumed_1'],
        # 'urban_times_consumed': (train['urban'] + 1) * train['svd_consumed_1'],
        # 'urban_times_sewer': train['urban'] * (train['sewer'] + 1),
        'consumed_per_hsize': train['svd_consumed_1'] / (train['hsize'] + 1),
        'infra_gap': train['svd_consumed_0'] - train['svd_infrastructure_0'],
        'worker_density': train['sfworkershh'] / (train['hsize'] + 1),
        'urban_sanitation': train['urban'] * train['sanitation_source'],
        # 'rural_nosewer': ((train.urban=='Rural') & (train.sewer=='No access')).apply(int),
        # 'old_and_low_family_size': train['hsize'] / train['age'],
        # 'age_per_hsize': train['age'] / (train['hsize'] + 1),
        # 'stable_workers': train['sfworkershh'] * train['sworkershh'] * (train['num_adult_male'] + train['num_adult_female']),
        # 'edu_potential_diff': train['educ_max'] - _sector_edu_mean,
        # 'dependency_interaction': (train['num_children5'] + train['num_children10'] + train['num_elderly']) / (train['hsize'] + 1),
        # 'dependency_ratio': (train['num_children5'] + train['num_children10'] + train['num_children18'] + train['num_elderly']) / (train['num_adult_male'] + train['num_adult_female'] + 1e-6),
        # 'adult_ratio': (train['num_adult_male'] + train['num_adult_female']) / (train['hsize'] + 1e-6),
        'rel_consumed_to_strata': train['svd_consumed_0'] / (_strata_mean + 1e-6),
        'diff_consumed_to_strata': train['svd_consumed_0'] - (_strata_mean + 1e-6),
        'zscore_consumed_to_strata': (train['svd_consumed_0'] - (_strata_mean + 1e-6)) / (_strata_std + 1e-6),
        'concat_consumed': train[['consumed3100', 'consumed1500', 'consumed2000', 'consumed3000', 'consumed1800', 'consumed3100']].apply(
        lambda x: sum([_val for _val in x]), axis=1),
        'lower_than_not_have_consumed': train[
            ['consumed900', 'consumed4100', 'consumed300',]].apply(
            lambda x: sum([_val for _val in x]), axis=1),
        # 'false_high_condition': (train['strata'].isin([1, 2, 3])) & (train['educ_max'] == 6) & (train['sector1d'].isin([0, 4, 6, 12])),
        # 'lower_than_and_no_access_not_have_consumed': train[
        #     ['consumed200', 'consumed900', 'consumed3100', 'region5']].apply(
        #     lambda x: x.region5 * 10 + sum([_val for _val in x[['consumed200', 'consumed900', 'consumed3100']]]), axis=1),
        'exp_per_hsize': train['utl_exp_ppp17'] / train['hsize'],
        'any_nonagoric_and_sewer': (train['any_nonagric'] + train['sewer']) / 2,
        # 'concat_consumed': train[
        #     ['consumed3100', 'consumed1500', 'consumed2000', 'consumed3000', 'consumed1800', 'consumed3100']].apply(
        #     lambda x: int(''.join([str(_val) for _val in x]), 2), axis=1),
        'has_child': train['has_child'].astype(int),
        # 'consumed_times_infra': train['svd_consumed_0'] * train['svd_infrastructure_0'],
        # 'edu_labor_efficiency': train['educ_max'] / (train['sector1d'] + 1),
        # 'utl_per_ae': train['utl_exp_ppp17'] / _adult_equivalence
        # 'infra_rel_to_strata': train['svd_infrastructure_0'] / (_infra_strata_mean + 1e-6),
        # 'infra_diff_to_strata': train['svd_infrastructure_0'] - _infra_strata_mean,
        # 'infra_zscore_to_strata': (train['svd_infrastructure_0'] - _infra_strata_mean) / (_infra_strata_mean + 1e-6)
        # 'nonagric_efficiency': train['any_nonagric'] * train['sfworkershh']
        # 'edu_diff_strata': train['educ_max'] - _strata_edu_mean
        # 'living_standard_index': train['svd_consumed_0'] * train['svd_infrastructure_0']
        # 'consumed_variety': train[_consumed_cols].sum(axis=1)
        # 'cons_z_in_sector': (train['svd_consumed_0'] - _sector_mean) / (_sector_std + 1e-6)
        # 'burden_factor': (train['hsize'] - train['sfworkershh']) / (train['sfworkershh'] + 1),
        # 'infra_cons_ratio': train['svd_infrastructure_0'] / (train['svd_consumed_0'] + 1e-6),
        # 'is_high_educ': (train['educ_max'] >= 5).astype(int)
        # 'urban_sector_combo': train['urban'] * 10 + train['sector1d']
        # 'modern_score': (
        #     (train['sanitation_source'] > 0).astype(int) +
        #     (train['sewer'] > 0).astype(int) +
        #     (train['any_nonagric'] > 0).astype(int)
        # ),
        # 'rel_wealth_vs_cons': train['relative_infra_wealth'] / (train['svd_consumed_0'] + 1e-6)
        # 'household_maturity': (train['educ_max'] * train['sfworkershh']) / (train['hsize'] + 1e-6)
    }

    return pd.DataFrame(_complex_numbers)


def survey_related_features(train: pd.DataFrame) -> pd.DataFrame:
        target_cols = [
            'svd_consumed_0', 'utl_exp_ppp17', 'sanitation_and_consumed',
            'sanitation_source', 'consumed_per_hsize', 'worker_density'
        ]

        df = train.copy()
        _latest_cols = df.columns
        for col in target_cols:
            _survey_group_mean = df.groupby('survey_id')[col].transform('mean')

            df[f'{col}_diff_survey'] = df[col] - _survey_group_mean
            df[f'{col}_ratio_survey'] = df[col] / (_survey_group_mean + 1e-6)
            df[f'{col}_rank_survey'] = df.groupby('survey_id')[col].rank(pct=True)
        return df.drop(columns=_latest_cols)


def target_encode(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series, cols: list[str], smoothing: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Target Encoding to categorical columns.

    Args:
        train: Training features
        test: Testing features
        target: Training target values
        cols: Columns to encode
        smoothing: Smoothing factor for the mean

    Returns:
        Encoded train and test DataFrames
    """
    train_encoded = train.copy()
    test_encoded = test.copy()

    prior = target.mean()

    for col in cols:
        stats = target.groupby(train[col]).agg(['count', 'mean'])
        counts = stats['count']
        means = stats['mean']

        smooth = (counts * means + smoothing * prior) / (counts + smoothing)

        train_encoded[col] = train[col].map(smooth)
        test_encoded[col] = test[col].map(smooth).fillna(prior)

    return train_encoded[cols], test_encoded[cols]


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
            'Piped water into dwelling': 7,
            'Surface water': 1,
            'Other': 2,
            'Piped water to yard/plot': 6,
            'Protected dug well': 5,
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
        'consumed100': _yes_no, 'consumed200': _yes_no, 'consumed300': _yes_no, 'consumed400': _yes_no,
        'consumed500': _yes_no, 'consumed600': _yes_no, 'consumed700': _yes_no, 'consumed800': _yes_no,
        'consumed900': _yes_no, 'consumed1000': _yes_no, 'consumed1100': _yes_no, 'consumed1200': _yes_no,
        'consumed1300': _yes_no, 'consumed1400': _yes_no, 'consumed1500': _yes_no, 'consumed1600': _yes_no,
        'consumed1700': _yes_no, 'consumed1800': _yes_no, 'consumed1900': _yes_no, 'consumed2000': _yes_no,
        'consumed2100': _yes_no, 'consumed2200': _yes_no, 'consumed2300': _yes_no, 'consumed2400': _yes_no,
        'consumed2500': _yes_no, 'consumed2600': _yes_no, 'consumed2700': _yes_no, 'consumed2800': _yes_no,
        'consumed2900': _yes_no, 'consumed3000': _yes_no, 'consumed3100': _yes_no, 'consumed3200': _yes_no,
        'consumed3300': _yes_no, 'consumed3400': _yes_no, 'consumed3500': _yes_no, 'consumed3600': _yes_no,
        'consumed3700': _yes_no, 'consumed3800': _yes_no, 'consumed3900': _yes_no, 'consumed4000': _yes_no,
        'consumed4100': _yes_no, 'consumed4200': _yes_no, 'consumed4300': _yes_no, 'consumed4400': _yes_no,
        'consumed4500': _yes_no, 'consumed4600': _yes_no, 'consumed4700': _yes_no, 'consumed4800': _yes_no,
        'consumed4900': _yes_no, 'consumed5000': _yes_no,
    }

    category_cols = [
        'water', 'toilet', 'sewer', 'elect', 'water_source',
        'male', 'urban',
        'owner', 'employed', 'any_nonagric',
        'sanitation_source', 'dweltyp', 'educ_max', 'sector1d',
        'region1', 'region2', 'region3', 'region4', 'region5', 'region6', 'region7',
        'consumed100', 'consumed200', 'consumed300', 'consumed400',
        'consumed500', 'consumed600', 'consumed700', 'consumed800',
        'consumed900', 'consumed1000', 'consumed1100', 'consumed1200',
        'consumed1300', 'consumed1400', 'consumed1500', 'consumed1600',
        'consumed1700', 'consumed1800', 'consumed1900', 'consumed2000',
        'consumed2100', 'consumed2200', 'consumed2300', 'consumed2400',
        'consumed2500', 'consumed2600', 'consumed2700', 'consumed2800',
        'consumed2900', 'consumed3000', 'consumed3100', 'consumed3200',
        'consumed3300', 'consumed3400', 'consumed3500', 'consumed3600',
        'consumed3700', 'consumed3800', 'consumed3900', 'consumed4000',
        'consumed4100', 'consumed4200', 'consumed4300', 'consumed4400',
        'consumed4500', 'consumed4600', 'consumed4700', 'consumed4800',
        'consumed4900', 'consumed5000',
    ]
    #
    # _custom_maps = {
    #     'strata_urban': {
    #         '1_0': 0, '1_1': 1,
    #         '2_0': 2, '2_1': 3,
    #         '3_0': 4, '3_1': 5,
    #         '4_0': 6, '4_1': 7,
    #         '5_0': 8, '5_1': 9,
    #         '6_0': 10, '6_1': 11,
    #         '7_0': 12, '7_1': 13,
    #         '8_0': 14, '8_1': 15,
    #     },
    #     'strata_sanitation': {
    #         '1_0': 0, '1_1': 1, '1_2': 2, '1_3': 3, '1_4': 4, '1_5': 5,
    #         '2_0': 6, '2_1': 7, '2_2': 8, '2_3': 9, '2_4': 10, '2_5': 11,
    #         '3_0': 12, '3_1': 13, '3_2': 14, '3_3': 15, '3_4': 16, '3_5': 17,
    #         '4_0': 18, '4_1': 19, '4_2': 20, '4_3': 21, '4_4': 22, '4_5': 23,
    #         '5_0': 24, '5_1': 25, '5_2': 26, '5_3': 27, '5_4': 28, '5_5': 29,
    #         '6_0': 30, '6_1': 31, '6_2': 32, '6_3': 33, '6_4': 34, '6_5': 35,
    #         '7_0': 36, '7_1': 37, '7_2': 38, '7_3': 39, '7_4': 40, '7_5': 41,
    #         '8_0': 42, '8_1': 43, '8_2': 44, '8_3': 45, '8_4': 46, '8_5': 47,
    #     }
    # }

    x_train = train.copy()
    _imputer = SimpleImputer(strategy='most_frequent')

    for _col in tqdm.tqdm(category_cols):

        _nulls = x_train[_col].isnull()
        x_train.loc[~_nulls, _col] = x_train.loc[~_nulls, _col].apply(lambda x: _category_number_maps[_col][x])
        x_train[_col] = pd.Series(map(round, _imputer.fit_transform(x_train[_col].to_numpy().reshape(-1, 1)).flatten()),
                                  index=x_train.index).astype(int)

    # x_train['strata_urban'] = (x_train['strata'].astype(str) + "_" + x_train['urban'].astype(str)).apply(lambda x: _custom_maps['strata_urban'][x])
    # x_train['strata_sanitation'] = (x_train['strata'].astype(str) + "_" + x_train['sanitation_source'].astype(str)).apply(lambda x: _custom_maps['strata_sanitation'][x]) / 2

    return x_train[category_cols].to_numpy(), category_cols


def create_new_features_array(df: pd.DataFrame) -> np.ndarray:

    _features, _ = _create_infra_features(df)

    return _features


def create_survey_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregates at the survey_id level.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with survey-level aggregated features
    """
    # Define columns to aggregate
    num_cols = ['hsize', 'age', 'num_children5', 'num_children10', 'num_children18',
                'num_adult_female', 'num_adult_male', 'num_elderly', 'sworkershh', 'sfworkershh']

    # Group by survey_id and calculate mean, std, etc.
    survey_groups = df.groupby('survey_id')[num_cols]

    means = survey_groups.transform('mean').add_suffix('_survey_mean')
    stds = survey_groups.transform('std').add_suffix('_survey_std')

    # Calculate ratios of household value to survey mean
    ratios = (df[num_cols] / means.to_numpy()).add_suffix('_to_survey_mean')

    return pd.concat([means, stds, ratios], axis=1)


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
