__all__ = ['validation_plot_parameters']

import matplotlib.pyplot as plt
import numpy as np

from ._common import fit_and_test_model


def validation_plot_parameters(model_name: str):
    _model_params, _scoring, _cv_params, _param_scales = _get_validation_params(model_name)
    _scoring = 'competition_score'

    # Validation curve - NO fit_params!
    for k, v in _cv_params.items():
        print(f'\nProcessing parameter: {k}')

        _train_scores, _valid_scores = [], []
        for _val in v:
            _params = _model_params.copy()
            _params[k] = _val
            _train_scores_per_survey_group, _valid_scores_per_survey_group = fit_and_test_model(model_names=[model_name], model_params=_params, display_result=False)
            _train_scores.append([_scores[_scoring] for _scores in _train_scores_per_survey_group])
            _valid_scores.append([_scores[_scoring] for _scores in _valid_scores_per_survey_group])

        _train_scores = np.array(_train_scores)
        _valid_scores = np.array(_valid_scores)
        print(f'{k}: {v}')
        print(f'Shape: {_train_scores.shape}, {_valid_scores.shape}')
        print(f'Valid scores - min: {np.min(_valid_scores):.3f}, max: {np.max(_valid_scores):.3f}')

        # Calculate statistics
        train_means = np.mean(_train_scores, axis=1)
        train_stds = np.std(_train_scores, axis=1)
        valid_means = np.mean(_valid_scores, axis=1)
        valid_stds = np.std(_valid_scores, axis=1)

        # Plot
        plt.figure(figsize=(10, 6))
        for train_mean, train_std, valid_mean, valid_std in zip(train_means, train_stds, valid_means, valid_stds):
            plt.plot(v, train_mean, 'o-', color='blue', label='Training score')
            plt.fill_between(v, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
            plt.plot(v, valid_mean, 's--', color='green', label='CV score')
            plt.fill_between(v, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2, color='green')

        # Set scale (skip log scale for gamma=0)
        if _param_scales[k] == 'log' and min(v) > 0:
            plt.xscale('log')

        plt.xlabel(k, fontsize=12)
        plt.ylabel(_scoring, fontsize=12)
        plt.title(f'Validation Curve: {k}', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        break


def _get_validation_params(model_name: str):
    # TODO: switch params per model_name
    model_params = dict(
        booster='gbtree',
        objective='reg:squarederror',
        random_state=42,
        n_estimators=100
    )

    # Cross-validation setup
    scoring = 'neg_mean_squared_error'

    # Fixed parameter ranges
    cv_params = {
        'subsample': [0.3, 0.5, 0.7, 0.9, 1.0],
        'colsample_bytree': [0.3, 0.5, 0.7, 0.9, 1.0],
        'reg_alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'reg_lambda': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'learning_rate': [0.001, 0.01, 0.1, 0.3],
        'min_child_weight': [1, 3, 5, 7, 10],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'gamma': [0, 0.001, 0.01, 0.1, 1.0]
    }

    param_scales = {
        'subsample': 'linear',
        'colsample_bytree': 'linear',
        'reg_alpha': 'log',
        'reg_lambda': 'log',
        'learning_rate': 'log',
        'min_child_weight': 'linear',
        'max_depth': 'linear',
        'gamma': 'log'
    }

    return model_params, scoring, cv_params, param_scales
