__all__ = ['param_compare']

import os
import pandas as pd
import matplotlib.pyplot as plt


def param_compare(
        df_a: pd.DataFrame, df_b: pd.DataFrame,
        labels: list[str], prefix: str, dir: str = '../plots',
        allow_overwrite: bool = False,
):
    os.makedirs(dir, exist_ok=allow_overwrite)

    for _col in set(df_a.columns) - {'hhid', 'com'}:

        if isinstance(df_a[_col], pd.Categorical) or pd.api.types.is_object_dtype(df_a[_col]):
            df_a_counts = df_a[_col].value_counts(normalize=True).sort_index(ascending=False,)
            df_b_counts = df_b[_col].value_counts(normalize=True).sort_index(ascending=False,)

            pd.DataFrame({labels[0]: df_a_counts, labels[1]: df_b_counts}).plot(kind='bar')
            if len(df_a_counts) == 2:
                plt.ylim(0, 1)

        else:
            plt.hist(df_a[_col], label=labels[0], density=True)
            plt.hist(df_b[_col], label=labels[1], density=True, alpha=0.7)
            plt.legend()

        plt.title(_col)
        plt.savefig(f'{dir}/{prefix}_{_col}.png', bbox_inches='tight')
        plt.close()