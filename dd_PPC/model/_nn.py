__all__ = ['get_tabnet_regressor', 'get_tabular_nn_regressor', 'get_mlp_nn_regressor', 'TabularNN', 'EntityEmbeddingMLP', 'Float32Transformer']

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import inspect
from collections.abc import Mapping
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, is_regressor
from pytorch_tabnet.tab_model import TabNetRegressor
from skorch import NeuralNetRegressor

from .._config import CATEGORY_NUMBER_MAPS, NUMBER_COLUMNS
from ..preprocess import complex_numbers_dataframe
from ..calc import CustomCompetitionLoss


def get_tabnet_regressor(params: dict) -> BaseEstimator:

    params = {} if params is None else params.copy()

    if 'seed' in params:
        torch.manual_seed(params['seed'])
        del params['seed']

    # Split constructor params vs fit-time params to avoid unexpected kwargs errors.
    _init_keys = set(inspect.signature(TabNetRegressor.__init__).parameters.keys()) - {'self', 'kwargs'}
    init_params = {k: v for k, v in params.items() if k in _init_keys}
    _fit_keys = {
        "X_train", "y_train", "eval_set", "eval_name", "eval_metric", "loss_fn", "weights",
        "max_epochs", "patience", "batch_size", "virtual_batch_size", "num_workers", "drop_last",
        "callbacks", "pin_memory", "from_unsupervised", "warm_start", "augmentations", "compute_importance"
    }
    fit_params = {k: v for k, v in params.items() if k in _fit_keys}

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    return _TabNetSklearnWrapper(
        init_params=dict(
            # Preprocessing already encodes categorical values; treat everything as continuous.
            cat_idxs=[],
            cat_dims=[],
            optimizer_fn=torch.optim.Adam,
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            device_name=device,
            **init_params
        ),
        fit_params=fit_params
    )


class _TabNetSklearnWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-friendly wrapper that handles y shape and fit-time kwargs for TabNet."""

    def __init__(self, init_params: dict | None = None, fit_params: dict | None = None):
        self.init_params = init_params or {}
        self.fit_params = fit_params if isinstance(fit_params, Mapping) else (fit_params or {})
        self._model = None
        self._estimator_type = "regressor"

    def fit(self, X, y, **kwargs):
        if y is not None and getattr(y, "ndim", 1) == 1:
            y = y.reshape(-1, 1)
        merged = {**self.fit_params, **kwargs}
        self._model = TabNetRegressor(**self.init_params)
        self._model.fit(X, y, **merged)
        return self

    def predict(self, X):
        preds = self._model.predict(X)
        if getattr(preds, "ndim", 1) == 2 and preds.shape[1] == 1:
            return preds[:, 0]
        return preds


def get_tabular_nn_regressor(params: dict) -> NeuralNetRegressor:
    """
    Creates and configures a NeuralNetRegressor for tabular data.

    This function initializes a NeuralNetRegressor using the listed parameters
    to handle tabular data tasks. It prepares continuous and categorical feature
    dimensions, defines embedding dimensions for categorical features, sets up
    loss and optimizer functions, and determines the appropriate device for
    training. The initialized regressor is then returned.

    Args:
        params (dict): A dictionary that holds the hyperparameters for the neural
            network. Acceptable keys include:
            - 'lr' (float): Learning rate for the optimizer. Default is 0.01.
            - 'max_epochs' (int): Maximum number of epochs for training. Default
              is 4.
            - 'batch_size' (int): Batch size for training data. Default is 32.
            - 'seed' (int, optional): Random seed for reproducibility. If provided,
              it will set the seed for PyTorch and will be removed from the params
              dict.

    Returns:
        NeuralNetRegressor: Configured neural network regressor for tabular data
        tasks.
    """

    if params is None:
        params = dict(lr=0.01, max_epochs=4, batch_size=32)

    if 'seed' in params:
        torch.manual_seed(params['seed'])
        del params['seed']

    num_features = len(NUMBER_COLUMNS)
    cat_features_dims = [len(m) + 1 for m in CATEGORY_NUMBER_MAPS.values()]
    emb_dims = [min(50, (d + 1) // 2) for d in cat_features_dims]

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    regressor = NeuralNetRegressor(
        module=TabularNN,
        module__n_cont=num_features,
        module__cat_dims=cat_features_dims,
        module__emb_dims=emb_dims,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=_device,
        **params
    )

    return regressor


def get_mlp_nn_regressor(params: dict) -> NeuralNetRegressor:
    """
    Creates and configures a multi-layer perceptron (MLP) based neural network regressor using PyTorch and the
    NeuralNetRegressor wrapper from the skorch library. This regressor leverages categorical embedding layers for
    handling categorical features alongside continuous features.

    Args:
        params (dict): A dictionary of parameters to configure the regressor. If not provided, default values
            are used. The dictionary may include:
            - 'lr' (float): Learning rate for the optimizer.
            - 'max_epochs' (int): Number of training epochs.
            - 'batch_size' (int): Size of the training batches.
            - 'seed' (int, optional): Random seed for reproducibility. If provided, it will set the PyTorch
              random seed and will be removed from the dictionary before configuring the regressor.

    Returns:
        NeuralNetRegressor: A configured NeuralNetRegressor instance, ready for training and evaluation.
    """

    if params is None:
        params = dict(lr=0.001, max_epochs=7, batch_size=32)

    if 'seed' in params:
        torch.manual_seed(params['seed'])
        del params['seed']

    _complex_input_cols = NUMBER_COLUMNS + ['svd_consumed_0', 'svd_infrastructure_0', 'urban', 'sanitation_source', 'svd_consumed_1']

    complex_output_cols = list(complex_numbers_dataframe(pd.DataFrame(
        [[0] * len(_complex_input_cols), [1] * len(_complex_input_cols)],
        columns=_complex_input_cols)).columns)
    svd_cols = [f'svd_consumed_{i}' for i in range(3)] + [f'svd_infrastructure_{i}' for i in range(3)]

    final_num_cols = NUMBER_COLUMNS + svd_cols + complex_output_cols

    num_features = len(final_num_cols)
    cat_features_dims = [len(m) + 1 for m in CATEGORY_NUMBER_MAPS.values()]
    emb_dims = [min(50, (d + 1) // 2) for d in cat_features_dims]

    regressor = NeuralNetRegressor(
        module=EntityEmbeddingMLP,
        module__n_cont=num_features,
        module__cat_dims=cat_features_dims,
        module__emb_dims=emb_dims,
        criterion=nn.SmoothL1Loss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        **params
    )

    return regressor


class TabularNN(nn.Module):
    def __init__(self, n_cont, cat_dims, emb_dims):
        super().__init__()
        self.n_cont = n_cont

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=d, embedding_dim=e)
            for d, e in zip(cat_dims, emb_dims)
        ])

        total_input_dim = n_cont + sum(emb_dims)
        self.fc = nn.Sequential(
            nn.Linear(total_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, X):
        # Xは (batch_size, features) の形
        # 1. 数値変数とカテゴリ変数を分離 (スライス)
        # 前半が数値変数、後半がカテゴリ変数と仮定（ColumnTransformerの順序）
        x_cont = X[:, :self.n_cont].float()
        x_cat = X[:, self.n_cont:].long()

        # 2. カテゴリ変数をEmbedding
        x_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(x_emb, dim=1)

        # 3. 結合して全結合層へ
        x = torch.cat([x_cont, x_emb], dim=1)

        return self.fc(x).squeeze(-1) # 出力を (batch_size,) に


class EntityEmbeddingMLP(nn.Module):
    def __init__(self, n_cont, cat_dims, emb_dims):
        super().__init__()
        self.n_cont = n_cont
        # 各カテゴリ変数ごとの埋め込み層
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=d, embedding_dim=e)
            for d, e in zip(cat_dims, emb_dims)
        ])

        # 入力サイズ = 数値変数の数 + 埋め込みベクトルの合計サイズ
        total_input_dim = n_cont + sum(emb_dims)

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        x_cont = X[:, :self.n_cont].float()
        x_cat = X[:, self.n_cont:].long()

        x_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(x_emb, dim=1)

        x = torch.cat([x_cont, x_emb], dim=1)

        return self.mlp(x).squeeze(-1)


class Float32Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):

        if isinstance(X, pd.DataFrame): X = X.to_numpy()

        return X.astype(np.float32)


class MTLConsPoverty(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, pov_dim: int = 19):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head_cons = nn.Linear(hidden, 1)
        self.head_pov = nn.Linear(hidden, pov_dim)

    def forward(self, x):
        h = self.shared(x)
        cons = self.head_cons(h).squeeze(-1)
        pov = torch.sigmoid(self.head_pov(h))
        return cons, pov


class MTLLossWrapper(nn.Module):
    def __init__(self, poverty_weights):
        super().__init__()
        self.loss = CustomCompetitionLoss(poverty_weights)

    def forward(self, outputs, targets):
        cons_pred, pov_pred = outputs
        cons_true, pov_true = targets
        return self.loss(cons_pred, cons_true, pov_pred, pov_true)
