__all__ = ['get_tabular_nn_regressor', 'TabularNN', 'Float32Transformer']

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from skorch import NeuralNetRegressor

from .._config import CATEGORY_NUMBER_MAPS, NUMBER_COLUMNS


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

    regressor = NeuralNetRegressor(
        module=TabularNN,
        module__n_cont=num_features,
        module__cat_dims=cat_features_dims,
        module__emb_dims=emb_dims,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
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


class Float32Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):

        if isinstance(X, pd.DataFrame): X = X.to_numpy()

        return X.astype(np.float32)