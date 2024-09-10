import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def calculate_metrics(true: list[bool], pred: list[bool]):
    metrics = {
        'precision': precision_score(true, pred, zero_division=np.nan),
        'recall': recall_score(true, pred, zero_division=np.nan),
        'accuracy': accuracy_score(true, pred),
    }
    f1 = f1_score(true, pred, zero_division=np.nan)
    # replace 0.0 with NaN
    metrics['f1'] = np.nan if f1 == 0.0 else f1
    return metrics


def calculate_model_metrics(
    df: pd.DataFrame,
    models: list[str],
    standardized_symptoms: list[str],
    label_suffix: str = '_gs',
):
    all_metrics = {}

    for model in models:
        model_metrics = {}
        for symptom in standardized_symptoms:
            # Retrieve the true and predicted values
            true_series = df[f'{symptom}{label_suffix}']
            pred_series = df[f'{symptom}_{model}']

            # Create mask based on NaN values in pred_series
            mask = ~pred_series.isna()

            # Apply the mask to both true and pred before dropping NaNs and converting types
            true_masked = true_series[mask].astype('bool')
            pred_masked = pred_series[mask].astype('bool')

            # Calculate metrics
            symptom_metrics = calculate_metrics(true_masked, pred_masked)

            # Include the symptom as a key in the model_metrics dictionary
            model_metrics[symptom] = symptom_metrics

        # Include the model as a key in the all_metrics dictionary
        all_metrics[model] = model_metrics

    return all_metrics


def metrics_to_df(model_metrics, model_name):
    df = pd.DataFrame(model_metrics).T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'symptom'}, inplace=True)
    df['model'] = model_name
    return df


def combine_metrics_df(all_metrics, models):
    dfs = [metrics_to_df(all_metrics[model], model) for model in models]
    combined_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    return combined_df


def bootstrap_metrics(df, true_col, pred_col, n_iterations=1000, confidence_level=0.95, deterministic: bool = False):
    metrics = {
        'precision': [],
        'recall': [],
        'accuracy': [],
        'f1': []
    }

    n = len(df)

    for i in range(n_iterations):
        random_state = 42 + i if deterministic else None
        bootstrap_sample = df.sample(n=n, replace=True, random_state=random_state)
        true_values = bootstrap_sample[true_col].astype(bool)
        pred_values = bootstrap_sample[pred_col].astype(bool)

        sample_metrics = calculate_metrics(true_values, pred_values)

        for metric in metrics:
            metrics[metric].append(sample_metrics[metric])

    ci_bounds = {}
    for metric, values in metrics.items():
        mean = np.mean(values)
        lower_bound = np.percentile(values, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(values, (1 + confidence_level) / 2 * 100)
        ci_bounds[metric] = (mean, lower_bound, upper_bound)

    return ci_bounds


def calculate_model_metrics_with_bootstrap(
    df, models, standardized_symptoms, n_iterations=1000, confidence_level=0.95, deterministic: bool = False
):
    all_metrics = {}

    for model in models:
        model_metrics = {}
        for symptom in standardized_symptoms:
            true_col = f'{symptom}_gs'
            pred_col = f'{symptom}_{model}'

            # Retrieve the true and predicted values
            true_series = df[true_col]
            pred_series = df[pred_col]

            # Create mask based on NaN values in pred_series
            mask = ~pred_series.isna()

            # Apply the mask to both true and pred before dropping NaNs and converting types
            true_masked = true_series[mask].astype('bool')
            pred_masked = pred_series[mask].astype('bool')

            # Combine masked true and pred into a DataFrame for bootstrap function
            combined_df = pd.DataFrame({true_col: true_masked, pred_col: pred_masked})

            # Calculate bootstrap metrics
            symptom_metrics = bootstrap_metrics(combined_df, true_col, pred_col, n_iterations, confidence_level, deterministic)

            # Include the symptom as a key in the model_metrics dictionary
            model_metrics[symptom] = symptom_metrics

        # Include the model as a key in the all_metrics dictionary
        all_metrics[model] = model_metrics

    return all_metrics


def bootstrapped_metrics_to_df(model_metrics, model_name):
    records = []
    for symptom, metrics in model_metrics.items():
        for metric, (mean, lower_bound, upper_bound) in metrics.items():
            records.append({
                'symptom': symptom,
                'metric': metric,
                'mean': mean,
                '95% CI lower': lower_bound,
                '95% CI upper': upper_bound,
                'model': model_name
            })
    return pd.DataFrame(records)


def combine_bootstrapped_metrics_df(all_metrics, models):
    dfs = [bootstrapped_metrics_to_df(all_metrics[model], model) for model in models]
    combined_df = pd.concat(dfs, axis=0)
    return combined_df


def create_symptom_df_from_df(df):
    grouped = df.groupby(['symptom', 'model'])
    rows = []

    for (symptom, model), group in grouped:
        precision_row = group[group['metric'] == 'precision'].iloc[0]
        recall_row = group[group['metric'] == 'recall'].iloc[0]
        accuracy_row = group[group['metric'] == 'accuracy'].iloc[0]
        f1_row = group[group['metric'] == 'f1'].iloc[0]

        row = {
            "Symptom": symptom,
            "Model": model,
            "Precision": f"{precision_row['mean']:.2f} ({precision_row['95% CI lower']:.2f}–{precision_row['95% CI upper']:.2f})" if not pd.isna(precision_row['mean']) else '',
            "Recall": f"{recall_row['mean']:.2f} ({recall_row['95% CI lower']:.2f}–{recall_row['95% CI upper']:.2f})" if not pd.isna(recall_row['mean']) else '',
            "Accuracy": f"{accuracy_row['mean']:.2f} ({accuracy_row['95% CI lower']:.2f}–{accuracy_row['95% CI upper']:.2f})" if not pd.isna(accuracy_row['mean']) else '',
            "F1 Score": f"{f1_row['mean']:.2f} ({f1_row['95% CI lower']:.2f}–{f1_row['95% CI upper']:.2f})" if not pd.isna(f1_row['mean']) else ''
        }
        rows.append(row)

    result_df = pd.DataFrame(rows)
    return result_df
