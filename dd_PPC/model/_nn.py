import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin

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