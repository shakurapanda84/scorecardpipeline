import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from scipy.stats import ks_2samp
from typing import Tuple
import matplotlib.pyplot as plt

def calculate_auc(y_true, y_scores):
    """Calculate the Area Under the ROC Curve (AUC)."""
    return roc_auc_score(y_true, y_scores)

def calculate_gini(y_true, y_scores):
    """Calculate the Gini coefficient."""
    auc = calculate_auc(y_true, y_scores)
    return 2 * auc - 1

def calculate_accuracy_ratio(y_true, y_scores):
    """Calculate the Accuracy Ratio."""
    gini = calculate_gini(y_true, y_scores)
    return gini 

def calculate_confusion_matrix_metrics(y_true, y_pred):
    """Calculate confusion matrix metrics: precision, recall, and F1 score."""
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def calculate_ks(y_true, y_scores):
    """Calculate the KS statistic."""
    return ks_2samp(y_scores[y_true == 1], y_scores[y_true == 0]).statistic

def calculate_psi(expected, actual, bins=10):
    """Calculate the Population Stability Index (PSI)."""
    expected_bins = pd.cut(expected, bins=bins, retbins=True)[1]
    expected_counts = np.histogram(expected, bins=expected_bins)[0]
    actual_counts = np.histogram(actual, bins=expected_bins)[0]

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = np.sum(psi_values)

    return psi

def calculate_csi(expected, actual, bins=10):
    """Calculate the Characteristic Stability Index (CSI)."""
    # CSI is similar to PSI but often used for categorical variables
    return calculate_psi(expected, actual, bins)

def calculate_p_value(y_true, y_scores):
    """Calculate the p-value for the KS test."""
    ks_statistic, p_value = ks_2samp(y_scores[y_true == 1], y_scores[y_true == 0])
    return p_value 

def calculate_csi_v2(baseline_sample: pd.DataFrame,
                    validation_sample: pd.DataFrame,
                    features: list,
                    bins: int = 10,
                    categorical_features: list = None) -> dict:
    """
    Calculate Characteristic Stability Index (CSI) for multiple features.
    Similar to PSI but calculated at feature level.
    
    Parameters:
    -----------
    baseline_sample : pd.DataFrame
        Reference/baseline sample
    validation_sample : pd.DataFrame
        Sample to compare against baseline
    features : list
        List of feature names to calculate CSI for
    bins : int, default=10
        Number of bins for continuous variables
    categorical_features : list, default=None
        List of categorical features that don't need binning
        
    Returns:
    --------
    dict
        Dictionary containing CSI values and details for each feature
        
    Notes:
    ------
    CSI interpretation:
    < 0.1: No significant change
    0.1 - 0.2: Moderate change
    > 0.2: Significant change
    """
    
    if categorical_features is None:
        categorical_features = []
        
    results = {}
    
    for feature in features:
        is_categorical = feature in categorical_features
        
        try:
            # Calculate CSI for each feature
            csi_value, csi_details = _calculate_single_csi(
                baseline_sample[feature],
                validation_sample[feature],
                bins=bins if not is_categorical else None,
                is_categorical=is_categorical
            )
            
            results[feature] = {
                'csi': csi_value,
                'details': csi_details,
                'status': _get_csi_status(csi_value),
                'is_categorical': is_categorical
            }
            
        except Exception as e:
            results[feature] = {
                'csi': None,
                'details': f"Error: {str(e)}",
                'status': 'ERROR',
                'is_categorical': is_categorical
            }
            
    return results

def _calculate_single_csi(baseline_series: pd.Series,
                         validation_series: pd.Series,
                         bins: int = None,
                         is_categorical: bool = False) -> Tuple[float, pd.DataFrame]:
    """
    Calculate CSI for a single feature
    """
    if is_categorical:
        # For categorical variables, use value counts
        baseline_dist = baseline_series.value_counts(normalize=True)
        validation_dist = validation_series.value_counts(normalize=True)
        
        # Ensure both distributions have the same categories
        all_categories = sorted(set(baseline_dist.index) | set(validation_dist.index))
        baseline_dist = baseline_dist.reindex(all_categories, fill_value=0.0001)
        validation_dist = validation_dist.reindex(all_categories, fill_value=0.0001)
        
    else:
        # For continuous variables, create bins
        bin_edges = pd.qcut(
            pd.concat([baseline_series, validation_series]), 
            q=bins, 
            duplicates='drop',
            retbins=True
        )[1]
        
        # Calculate distributions
        baseline_dist = pd.cut(baseline_series, bins=bin_edges).value_counts(normalize=True)
        validation_dist = pd.cut(validation_series, bins=bin_edges).value_counts(normalize=True)
    
    # Create detailed DataFrame for the analysis
    details = pd.DataFrame({
        'Baseline_Dist': baseline_dist,
        'Validation_Dist': validation_dist
    })
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    details = details.fillna(epsilon)
    
    # Calculate CSI components
    details['Difference'] = details['Validation_Dist'] - details['Baseline_Dist']
    details['Log_Ratio'] = np.log(details['Validation_Dist'] / details['Baseline_Dist'])
    details['CSI_Component'] = details['Difference'] * details['Log_Ratio']
    
    # Calculate total CSI
    csi_value = details['CSI_Component'].sum()
    
    # Add contribution percentage
    details['Contribution_Pct'] = (details['CSI_Component'] / csi_value * 100).abs()
    
    return csi_value, details

def _get_csi_status(csi_value: float) -> str:
    """
    Get status based on CSI value
    """
    if csi_value < 0.1:
        return 'STABLE'
    elif csi_value < 0.2:
        return 'MODERATE_CHANGE'
    else:
        return 'SIGNIFICANT_CHANGE'

def analyze_csi_results(csi_results: dict,
                       output_path: str = None,
                       plot: bool = True) -> pd.DataFrame:
    """
    Analyze and visualize CSI results
    
    Parameters:
    -----------
    csi_results : dict
        Results from calculate_csi_v2
    output_path : str, optional
        Path to save visualizations
    plot : bool, default=True
        Whether to generate plots
        
    Returns:
    --------
    pd.DataFrame
        Summary of CSI analysis
    """
    # Create summary DataFrame
    summary = pd.DataFrame([
        {
            'Feature': feature,
            'CSI': results['csi'],
            'Status': results['status'],
            'Is_Categorical': results['is_categorical']
        }
        for feature, results in csi_results.items()
        if results['csi'] is not None
    ])
    
    if plot:
        # 1. CSI Values Plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(summary['Feature'], summary['CSI'])
        plt.axhline(y=0.1, color='g', linestyle='--', label='Stable Threshold')
        plt.axhline(y=0.2, color='r', linestyle='--', label='Significant Change Threshold')
        
        # Color bars based on status
        colors = {'STABLE': 'green', 'MODERATE_CHANGE': 'yellow', 'SIGNIFICANT_CHANGE': 'red'}
        for bar, status in zip(bars, summary['Status']):
            bar.set_color(colors[status])
            
        plt.xticks(rotation=45, ha='right')
        plt.title('CSI Values by Feature')
        plt.ylabel('CSI Value')
        plt.legend()
        plt.tight_layout()
        
        if output_path:
            plt.savefig(f"{output_path}/csi_values.png")
        plt.close()
        
        # 2. Distribution Comparison Plots
        for feature, results in csi_results.items():
            if results['csi'] is not None:
                details = results['details']
                
                plt.figure(figsize=(10, 6))
                x = range(len(details))
                width = 0.35
                
                plt.bar([i - width/2 for i in x], details['Baseline_Dist'], 
                       width, label='Baseline', alpha=0.6)
                plt.bar([i + width/2 for i in x], details['Validation_Dist'], 
                       width, label='Validation', alpha=0.6)
                
                plt.title(f'Distribution Comparison - {feature}\nCSI: {results["csi"]:.4f}')
                plt.xlabel('Bins')
                plt.ylabel('Proportion')
                plt.legend()
                plt.tight_layout()
                
                if output_path:
                    plt.savefig(f"{output_path}/distribution_{feature}.png")
                plt.close()
    
    return summary

# Example usage
def demonstrate_csi_analysis():
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Baseline sample
    baseline = pd.DataFrame({
        'numeric_stable': np.random.normal(0, 1, n_samples),
        'numeric_drift': np.random.normal(0, 1, n_samples),
        'categorical_stable': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_drift': np.random.choice(['X', 'Y', 'Z'], n_samples, p=[0.6, 0.3, 0.1])
    })
    
    # Validation sample (with some drift)
    validation = pd.DataFrame({
        'numeric_stable': np.random.normal(0, 1, n_samples),
        'numeric_drift': np.random.normal(0.5, 1.5, n_samples),  # Changed mean and std
        'categorical_stable': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_drift': np.random.choice(['X', 'Y', 'Z'], n_samples, p=[0.2, 0.3, 0.5])  # Changed proportions
    })
    
    # Calculate CSI
    features = ['numeric_stable', 'numeric_drift', 'categorical_stable', 'categorical_drift']
    categorical_features = ['categorical_stable', 'categorical_drift']
    
    csi_results = calculate_csi_v2(
        baseline,
        validation,
        features=features,
        categorical_features=categorical_features
    )
    
    # Analyze results
    summary = analyze_csi_results(csi_results, plot=True)
    
    print("\nCSI Analysis Summary:")
    print("=====================")
    print(summary.to_string(index=False))
    
    return csi_results, summary

# Run demonstration
csi_results, summary = demonstrate_csi_analysis() 