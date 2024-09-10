import pandas as pd

from statsmodels.stats.contingency_tables import cochrans_q, mcnemar


def _add_true_if_correct_columns_to_df(df: pd.DataFrame, pred_prefix: str = 'pred_status_', label_column: str = 'label'):
    df_copy = df.copy()
    label_dtype = df_copy[label_column].dtype
    for col in df_copy.columns:
        if col.startswith(pred_prefix):
            if df_copy[col].dtype != label_dtype:
                raise ValueError(f'Different dtypes for {col} and {label_column}')
            model_name = col.split(pred_prefix)[1]
            df_copy[f'{model_name}_correct'] = df_copy[col] == df_copy[label_column]

    return df_copy


def cochrans_q_test(df: pd.DataFrame, pred_prefix: str = 'pred_status_', label_column: str = 'label'):
    q_df = _add_true_if_correct_columns_to_df(df, pred_prefix=pred_prefix, label_column=label_column)

    cols_for_test = [col for col in q_df.columns if col.endswith('_correct')]
    if len(cols_for_test) == 0:
        raise ValueError(f'No columns starting with {pred_prefix} found in the DataFrame')

    cochrans_q_array = q_df[cols_for_test].copy().astype(int).to_numpy()

    result = cochrans_q(cochrans_q_array)

    q_stat = result.statistic
    p_value = result.pvalue

    return q_stat, p_value


def mcnemar_pairwise(df: pd.DataFrame, pred_prefix: str = 'pred_status_', label_column: str = 'label'):
    q_df = _add_true_if_correct_columns_to_df(df, pred_prefix=pred_prefix, label_column=label_column)

    cols_for_test = [col for col in q_df.columns if col.endswith('_correct')]
    if len(cols_for_test) == 0:
        raise ValueError(f'No columns starting with {pred_prefix} found in the DataFrame')

    results = {}
    for i in range(len(cols_for_test)):
        for j in range(i + 1, len(cols_for_test)):
            col1 = cols_for_test[i]
            col2 = cols_for_test[j]
            contingency_table = pd.crosstab(q_df[col1], q_df[col2])

            # Extract counts
            b = contingency_table.loc[True, False] if (
                        True in contingency_table.index and False in contingency_table.columns) else 0
            c = contingency_table.loc[False, True] if (
                        False in contingency_table.index and True in contingency_table.columns) else 0

            # Construct the table for McNemar's test
            table = [[0, b], [c, 0]]

            # Perform McNemar's test
            result = mcnemar(table, exact=True)

            # Store the result
            results[f'{col1} vs {col2}'] = (result.statistic, result.pvalue)

    return results
