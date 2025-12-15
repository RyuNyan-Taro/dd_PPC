__all__ = ['weighted_average_of_consumption_and_poverty_rate', 'poverty_rates_from_consumption']

import numpy as np
import pandas as pd


def weighted_average_of_consumption_and_poverty_rate(
        consumption: pd.DataFrame,
        poverty_rate_pred: pd.DataFrame,
        poverty_rate_target: pd.DataFrame) -> float:
    """Calculate weighted average of consumption and poverty rate, which is the primary competition metric of the competition.

    Args:
        consumption: prediction and target values of consumption.
        poverty_rate_pred: Prediction values per survey id.
        poverty_rate_target: Target values per survey id.

    Returns:
        The weighted average of consumption and poverty rate.

    References:
        competition info: https://www.drivendata.org/competitions/305/competition-worldbank-poverty/page/965/

    """

    _consumption_weighted_averages = [_calc_consumption_weighted_average(_df['cons_pred'], _df['cons_ppp17']) for _, _df in consumption.groupby('survey_id')]
    _poverty_rate_weighted_averages = [_calc_poverty_rate_weighted_average(_pred.T.to_numpy(), poverty_rate_target.loc[_id, :].T.to_numpy()) for _id, _pred in poverty_rate_pred.groupby('survey_id')]

    _weighted_average = sum(_consumption_weighted_averages) + sum(_poverty_rate_weighted_averages)

    return _weighted_average / len(poverty_rate_pred)

# sub functions for weighted_average_of_consumption_and_poverty_rate
def _calc_consumption_weighted_average(preds: np.ndarray, targets: np.ndarray) -> float:
    _absolute_errors = [abs(_target - _pred) / _target for _pred, _target in zip(preds, targets)]

    return 10 / len(_absolute_errors) * sum(_absolute_errors)


def _calc_poverty_rate_weighted_average(preds: np.ndarray, targets: np.ndarray) -> float:
    _weights = [1 - abs(0.4 - _percent/100) for _percent in range(5, 100, 5)]
    _absolute_errors = [_weight * abs(_target - _pred) / _target for _pred, _target, _weight in zip(preds, targets, _weights)]

    return 90 / sum(_weights) * sum(_absolute_errors)


def poverty_rates_from_consumption(consumptions: pd.DataFrame, consumption_column: str = 'cons_ppp17') -> pd.DataFrame:
    _poverty_thresholds = [
        3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.70, 8.40, 9.13,
        9.87, 10.70, 11.62, 12.69, 14.03, 15.64, 17.76, 20.99, 27.37
    ]
    _columns = ['survey_id'] + [f'pct_hh_below_{_threshold}' for _threshold in _poverty_thresholds]

    _poverty_rates = []
    for _id, _df in consumptions.groupby('survey_id'):
        _poverty_rates.append([_id] + [np.sum(_df[consumption_column] < _threshold) / len(_df) for _threshold in _poverty_thresholds])

    return pd.DataFrame(_poverty_rates, columns=_columns)
