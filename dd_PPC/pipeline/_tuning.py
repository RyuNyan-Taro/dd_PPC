__all__ = ['validation_plot_parameters', 'tuning_model']

from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

from ._common import fit_and_test_model


def validation_plot_parameters(model_name: str, cv_params: dict | None = None):
    """Plot validation curves for hyperparameter tuning of a machine learning model.

    Args:
        model_name: learning model name
        cv_params: custom validation parameters

    References:
        Base idea and the code: https://qiita.com/c60evaporator/items/a9a049c3469f6b4872c6

    """
    _model_params, _scoring, _cv_params, _param_scales = _get_validation_params(model_name)

    if isinstance(cv_params, dict):
        _cv_params = cv_params
    _scoring = 'competition_score'

    # Validation curve - NO fit_params!
    for k, v in _cv_params.items():

        _train_scores, _valid_scores = [], []
        for _val_i, _val in enumerate(v, start=1):
            print(f'\nModel: {model_name}\nProcessing parameter: {k} {_val_i}/{len(v)} val: {_val}')

            _args = dict(
                model_names=[model_name], display_result=False
            )
            _params = _model_params.copy()
            if k == 'boxcox_lambda':
                _args['boxcox_lambda'] = _val
            else:
                _params[k] = _val
            _args['model_params'] = _params

            _train_scores_per_survey_group, _valid_scores_per_survey_group = fit_and_test_model(**_args)
            _train_scores.append([_scores[_scoring] for _scores in _train_scores_per_survey_group])
            _valid_scores.append([_scores[_scoring] for _scores in _valid_scores_per_survey_group])
            clear_output(wait=True)

        _train_scores = np.array(_train_scores)
        _valid_scores = np.array(_valid_scores)
        print(f'{k}: {v}')
        print(f'Shape: {_train_scores.shape}, {_valid_scores.shape}')
        print(f'Valid scores - min: {np.min(_valid_scores):.3f}, max: {np.max(_valid_scores):.3f}')

        print('train_scores:', _train_scores)
        print('valid_scores:', _valid_scores)

        # Plot
        fig, axes = plt.subplots(1, ncols=2, figsize=(10, 6), constrained_layout=True)

        _ax = axes[0]
        for _train, _valid, _c in zip(_train_scores.T, _valid_scores.T, ['gray', 'yellow', 'red']):
            _ax.plot(v, _train, 'o-', color=_c)
            _ax.plot(v, _valid, 's--', color=_c)
        _ax.set_ylabel(_scoring, fontsize=12)

        train_means = np.mean(_train_scores, axis=1)
        train_stds = np.std(_train_scores, axis=1)
        valid_means = np.mean(_valid_scores, axis=1)
        valid_stds = np.std(_valid_scores, axis=1)

        _ax = axes[1]
        _ax.plot(v, train_means, 'o-', color='blue', label='Training score')
        _ax.fill_between(v, train_means - train_stds, train_means + train_stds, alpha=0.2, color='blue')
        _ax.plot(v, valid_means, 's--', color='green', label='CV score')
        _ax.fill_between(v, valid_means - valid_stds, valid_means + valid_stds, alpha=0.2, color='green')
        _ax.legend(loc='upper left')

        # Set scale (skip log scale for gamma=0)
        if _param_scales[k] == 'log' and min(v) > 0:
            for _ax in axes:
                _ax.set_xscale('log')

        for _ax in axes:
            _ax.grid(True, alpha=0.3)

        fig.suptitle(f'Validation Curve: {k}', fontsize=14)

        plt.savefig(f'../plots/model_{model_name}_validation_curve_{k}.png', bbox_inches='tight')
        plt.show()


def tuning_model(model_name: str, n_trials: int = 100, timeout: int | None = None):
    """Hyperparameter tuning using Optuna.

    Args:
        model_name: Name of the model to tune ('xgboost', 'lightgbm', or 'catboost')
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds for optimization (None for no limit)

    Returns:
        best_params: Dictionary of best hyperparameters found
        best_score: Best competition score achieved
    """
    import optuna
    from optuna.samplers import TPESampler

    def objective(trial):
        """Optuna objective function."""

        # Define hyperparameter search space based on model
        if model_name == 'xgboost':
            params = {
                'booster': 'gbtree',
                'objective': 'reg:squarederror',
                'random_state': 123,
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 9),
                'gamma': trial.suggest_float('gamma', 0.1, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 1.0, log=True),
            }
        elif model_name == 'lightgbm':
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'force_row_wise': True,
                'random_state': 123,
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                'num_leaves': trial.suggest_int('num_leaves', 20, 256),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                # 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                # 'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            }
        elif model_name == 'catboost':
            params = {
                'boosting_type': 'Plain',
                'loss_function': 'RMSE',
                'random_state': 123,
                'verbose': False,
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'depth': trial.suggest_int('depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 9.0),
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Optionally tune boxcox_lambda
        # boxcox_lambda = trial.suggest_float('boxcox_lambda', 0.0, 0.3)

        # Run cross-validation with these parameters
        train_scores, test_scores = fit_and_test_model(
            model_names=[model_name],
            model_params=params,
            display_result=False
        )

        clear_output(wait=True)

        # Calculate mean competition score across folds
        mean_score = np.mean([score['competition_score'] for score in test_scores])

        # Optuna minimizes, but we want to minimize the competition score (lower is better)
        return mean_score

    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        study_name=f'{model_name}_tuning'
    )

    # Optimize
    print(f'Starting Optuna optimization for {model_name}...')
    print(f'Trials: {n_trials}, Timeout: {timeout}s\n')

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    # Results
    print('\n' + '=' * 60)
    print('Optimization Complete!')
    print('=' * 60)
    print(f'Best competition score: {study.best_value:.6f}')
    print(f'Best parameters:')
    for key, value in study.best_params.items():
        print(f'  {key}: {value}')

    # Save study results
    try:
        import joblib
        joblib.dump(study, f'../models/optuna_study_{model_name}.pkl')
        print(f'\nStudy saved to ../models/optuna_study_{model_name}.pkl')
    except Exception as e:
        print(f'\nWarning: Could not save study: {e}')

    return study.best_params, study.best_value


def _get_validation_params(model_name: str) -> tuple[dict, str, dict, dict]:
    # TODO: switch params per model_name
    model_params = {
        'xgboost': dict(
            booster='gbtree',
            objective='reg:squarederror',
            random_state=42,
            n_estimators=100
        ),
        'lightgbm': dict(
            boosting_type='gbdt',
            objective='regression',
            force_row_wise=True,
            random_state=42,
            n_estimators=100
        ),
        'catboost': dict(
            boosting_type='Plain',
            loss_function='RMSE',
            random_state=42,
            n_estimators=100
        )
    }[model_name]

    # Cross-validation setup
    scoring = 'neg_mean_squared_error'

    # Fixed parameter ranges
    param_and_scales = {
        'xgboost': [
            {
                'subsample': [0.3, 0.5, 0.7, 0.9, 1.0],
                'colsample_bytree': [0.3, 0.5, 0.7, 0.9, 1.0],
                'reg_alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'reg_lambda': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'learning_rate': [0.001, 0.01, 0.1, 0.3],
                'min_child_weight': [1, 3, 5, 7, 10],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'gamma': [0, 0.001, 0.01, 0.1, 1.0]
            },
            {
                'subsample': 'linear',
                'colsample_bytree': 'linear',
                'reg_alpha': 'log',
                'reg_lambda': 'log',
                'learning_rate': 'log',
                'min_child_weight': 'linear',
                'max_depth': 'linear',
                'gamma': 'log',
                'boxcox_lambda': 'linear'
            }
        ],
        'lightgbm': [
            {
                'reg_alpha': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                'reg_lambda': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                'num_leaves': [2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256],
                'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
                'min_child_samples': [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            },
            {
                'reg_alpha': 'log',
                'reg_lambda': 'log',
                'num_leaves': 'linear',
                'colsample_bytree': 'linear',
                'subsample': 'linear',
                'subsample_freq': 'linear',
                'min_child_samples': 'linear'
            },
        ],
        'catboost': [
            {
                'depth': [4, 7, 10],
                'learning_rate': [0.001, 0.01, 0.1, 0.3],
                'l2_leaf_reg': [1, 4, 9],
            },
            {
                'depth': 'linear',
                'learning_rate': 'log',
                'l2_leaf_reg': 'linear'
            }
        ]
    }

    cv_params, param_scales = param_and_scales[model_name]

    return model_params, scoring, cv_params, param_scales
