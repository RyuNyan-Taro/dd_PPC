__all__ = ['weighted_average_of_consumption_and_poverty_rate']

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
        survey_ids: The survey IDs of the training data.

    Returns:
        The weighted average of consumption and poverty rate.

    References:
        competition info: https://www.drivendata.org/competitions/305/competition-worldbank-poverty/page/965/

    """

    _consumption_weighted_averages = [_calc_consumption_weighted_average(_df['cons_pred'], _df['cons_ppp17']) for _, _df in consumption.groupby('survey_id')]
    _poverty_rate_weighted_averages = [_calc_poverty_rate_weighted_average(_pred.to_numpy(), poverty_rate_target[_id].T.to_numpy()) for _id, _pred in poverty_rate_pred.T.groupby('survey_id')]

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