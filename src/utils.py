import ast
import pandas as pd


def check_null_in_response_columns(df: pd.DataFrame, prefix: str = 'response_'):
    """
    Checks for null values in columns with specified prefix and prints the column name along with the number of null values.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    prefix (str): The prefix of the columns to check for null values.
    """
    for col in df.columns:
        if col.startswith(prefix):
            null_count = df[col].isnull().sum()
            print(f"Column '{col}' has {null_count} null values.")


def convert_status_response_columns(df: pd.DataFrame, prefix: str = 'response_status'):
    """
    Converts columns with specified prefix to boolean based on the presence of 'status' key in dictionary values
    and drops the original columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """

    columns_to_convert = [col for col in df.columns if col.startswith(prefix)]

    for col in columns_to_convert:
        new_col = 'pred_' + col[len(prefix):]
        df[new_col] = df[col].apply(lambda x: x.get('status') if isinstance(x, dict) and 'status' in x else None)

    df.drop(columns=columns_to_convert, inplace=True)
    return df


def standardize_name(name):
    """
    Standardizes a name by converting it to lowercase and replacing spaces and hyphens with underscores.

    :param name: The name to standardize.
    :return: The standardized name.
    """
    return name.strip().lower().replace(' ', '_').replace('-', '_')


def convert_detail_response_columns(
    df: pd.DataFrame,
    prefix: str = 'response_detail_',
    standardize_keys: bool = True,
):
    """
    Converts columns with specified prefix to separate columns based on dictionary values and drops the original columns.

    :param df:  The DataFrame to process.
    :param prefix:  The prefix of the columns to convert.
    :param standardize_keys:  Whether to standardize the keys of the dictionary values. Standardization involves converting
        the keys to lowercase and replacing spaces and hyphens with underscores. Default is True.
    :return:  The modified DataFrame.
    """
    new_df = df.copy().reset_index(drop=True)
    for col in new_df.columns:
        if col.startswith(prefix):
            model_name = col.split(prefix)[1]
            dict_values = new_df[col].map(
                lambda x: ast.literal_eval(x)
                if isinstance(x, str) else x
            )
            if standardize_keys:
                dict_values = dict_values.map(
                    lambda x: {standardize_name(k): v for k, v in x.items()}
                    if isinstance(x, dict) else x
                )
            expanded_values = pd.json_normalize(dict_values).add_suffix(f'_{model_name}')
            new_df = pd.concat([new_df, expanded_values], axis=1)
    return new_df


def convert_label_column(
    df: pd.DataFrame,
    label_column: str,
    allowed_symptoms: list[str],
    sep: str = ';',
    suffix: str = '_gs',
    standardize_keys: bool = True,
):
    """
    Converts a column with a list of symptoms to separate columns based on the allowed symptoms and adds a suffix to the column names.

    :param df:  The DataFrame to process.
    :param label_column:  The name of the column containing the list of symptoms.
    :param allowed_symptoms:  The list of allowed symptoms to create columns for.
    :param sep:  The separator used to split the list of symptoms. Default is ';'.
    :param suffix:  The suffix to add to the column names. Default is '_gs'.
    :param standardize_keys:  Whether to standardize the keys of the dictionary values. Standardization involves converting
        the keys to lowercase and replacing spaces and hyphens with underscores. Default is True.
    :return:  The modified DataFrame.
    """
    new_df = df.copy().reset_index(drop=True)

    if standardize_keys:
        allowed_symptoms = [standardize_name(symptom) for symptom in allowed_symptoms]

    # Initialize an empty list to hold all symptom dictionaries
    all_symptom_dicts = []

    for index, row in new_df.iterrows():
        symptom_dict = {
            symptom: False for symptom in allowed_symptoms
        }

        symptoms = row[label_column].split(sep)
        for symptom in symptoms:
            if standardize_keys:
                symptom = standardize_name(symptom)
            if symptom in symptom_dict:
                symptom_dict[symptom] = True

        # Append the dictionary to the list of all symptom dictionaries
        all_symptom_dicts.append(symptom_dict)

    # Convert the list of dictionaries into a DataFrame using json_normalize
    gs_symptoms_df = pd.json_normalize(all_symptom_dicts)

    # Add suffix '_gs' to each column
    gs_symptoms_df.columns = [f'{col}{suffix}' for col in gs_symptoms_df.columns]

    # Reset the index of both DataFrames before concatenating to ensure alignment
    new_df.reset_index(drop=True, inplace=True)
    gs_symptoms_df.reset_index(drop=True, inplace=True)

    # Concatenate the DataFrames along the columns axis
    return pd.concat([new_df, gs_symptoms_df], axis=1)
