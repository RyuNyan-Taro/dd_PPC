__all__ = ['param_compare']


def param_compare(df_a, df_b, labels: list[str], prefix: str, dir: str = '../plots'):
    for _col in set(df_a.columns) - {'hhid', 'com'}:

        if isinstance(df_a[_col], pd.Categorical) or pd.api.types.is_object_dtype(df_a[_col]):
            df_a_counts = df_a[_col].value_counts(normalize=True)
            df_b_counts = df_b[_col].value_counts(normalize=True)

            pd.DataFrame({labels[0]: df_a_counts, labels[1]: df_b_counts}).plot(kind='bar')
            if len(df_a_counts) == 2:
                plt.ylim(0, 1)

        else:
            plt.hist(df_a[_col], density=True, label=labels[0])
            plt.hist(df_b[_col], density=True, label=labels[1])
            plt.legend()

        plt.title(_col)
        plt.savefig(f'{dir}/{prefix}_{_col}.png', bbox_inches='tight')
        plt.close()