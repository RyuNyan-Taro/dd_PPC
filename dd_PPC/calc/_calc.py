__all__ = []

import numpy as np


def weighted_average_of_consumption_and_poverty_rate(
        capita_consumption_pred: np.ndarray, capita_consumption_target: np.ndarray,
        poverty_rate_pred: np.ndarray, poverty_rate_target: np.ndarray,
        survey_ids: np.ndarray) -> float:
    """Calculate weighted average of consumption and poverty rate, which is the primary competition metric of the competition.

    Args:
        capita_consumption_pred: Prediction values.
        capita_consumption_target: Target values.
        poverty_rate_pred: Prediction values.
        poverty_rate_target: Target values.
        survey_ids: The survey IDs of the training data.

    Returns:
        The weighted average of consumption and poverty rate.

    References:
        competition info: https://www.drivendata.org/competitions/305/competition-worldbank-poverty/page/965/

    """
    _ids = np.unique(survey_ids)

    _weighted_average = 0

    for _id in _ids:
        _cond = survey_ids == _id
        _consumption_weighted_average = _calc_consumption_weighted_average(capita_consumption_pred[_cond], capita_consumption_target[_cond])
        _poverty_rate_weighed_average = _calc_poverty_rate_weighted_average(poverty_rate_pred[_cond], poverty_rate_target[_cond])
        _weighted_average += _consumption_weighted_average + _poverty_rate_weighed_average

    return _weighted_average / len(_ids)


def _calc_consumption_weighted_average(pred: np.ndarray, target: np.ndarray) -> float:
    return 0


def _calc_poverty_rate_weighted_average(pred: np.ndarray, target: np.ndarray) -> float:
    return 0