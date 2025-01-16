import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import Dict
from scipy import signal

def calculate_cross_correlation(df, col1, col2, patient_id=None):
    """
    Calculate cross-correlation between two columns for a specific patient
    
    Parameters:
    df: DataFrame containing the data
    col1: First column name
    col2: Second column name
    patient_id: Optional - specific patient ID to analyze
    
    Returns:
    tuple: (correlation values, lags)
    """
    # If patient_id is specified, filter the dataframe
    if patient_id is not None:
        df = df[df['Patient ID'] == patient_id].copy()
    
    # Sort by Time(moment)
    df = df.sort_values('Time (moment)')
    
    # Special handling for RR_stationary
    if col2 == 'RR_stationary':
        # Calculate RR_stationary properly
        df['RR_stationary'] = df['RR interval msec'] - df['RR interval msec'].mean()
        series2 = df['RR_stationary']
    else:
        series2 = df[col2]
    
    # Get the time differences in milliseconds
    time_diffs = df['Time (moment)'].diff().dt.total_seconds() * 1000
    
    # Remove mean from both series
    series1 = df[col1] - df[col1].mean()
    series2 = series2 - series2.mean()
    
    # Calculate cross-correlation
    correlation = signal.correlate(series1, series2, mode='full')
    
    # Normalize
    correlation = correlation / (len(series1) * series1.std() * series2.std())
    
    # Calculate lags in milliseconds
    lags = signal.correlation_lags(len(series1), len(series2))
    lags = lags * time_diffs.mean()  # Convert lags to milliseconds
    
    return correlation, lags

def plot_cross_correlation(df, col1, col2, patient_id=None) -> bytes:
    """Return cross-correlation plot as bytes"""
    correlation, lags = calculate_cross_correlation(df, col1, col2, patient_id)
    
    plt.figure(figsize=(12, 6))
    plt.stem(lags, correlation)
    plt.xlabel('Lag (milliseconds)')
    plt.ylabel('Cross-correlation')
    title = f'Cross-correlation between {col1} and {col2}'
    if patient_id is not None:
        title += f' for Patient {patient_id}'
    plt.title(title, pad=30)
    plt.grid(True)
    
    # Find and plot the maximum correlation
    max_corr_idx = np.argmax(np.abs(correlation))
    max_lag = lags[max_corr_idx]
    max_corr = correlation[max_corr_idx]
    
    plt.plot(max_lag, max_corr, 'ro')
    plt.annotate(f'Max correlation: {max_corr:.3f}\nLag: {max_lag:.2f}ms',
                xy=(max_lag, max_corr),
                xytext=(10, 10),
                textcoords='offset points')
    
    # Save plot to bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    return buffer.getvalue()

def plot_patient_correlations(df: pd.DataFrame, patient_id: str) -> Dict[str, bytes]:
    """
    Generate all relevant correlation plots for a patient and return as bytes
    
    Parameters:
    df: DataFrame containing the data
    patient_id: ID of the patient to analyze
    
    Returns:
    Dict[str, bytes]: Dictionary mapping plot names to plot images as bytes
    """
    # Filter data for the specific patient
    patient_data = df[df['Patient ID'] == patient_id].copy()
    
    # Make RR interval stationary by taking first difference
    patient_data['RR_stationary'] = patient_data['RR interval msec'].diff()
    
    # Define correlations to plot
    mr_correlations = [
        'LA area cm2',
        'LA length cm',
        'MV tenting height mm',
        'MV annulus mm',
        'LV length cm',
        'LV area cm2',
        'RR_stationary'
    ]
    
    correlation_plots = {}

    # Generate plots for MR area correlations
    for col in mr_correlations:
        plot_name = f'patient_{patient_id}_MR_area_vs_{col.replace(" ", "_")}'
        correlation_plots[plot_name] = plot_cross_correlation(
            patient_data, 'MR area cm2', col, patient_id
        )

    # Generate plots for MR VC correlations
    for col in mr_correlations:
        plot_name = f'patient_{patient_id}_MR_VC_vs_{col.replace(" ", "_")}'
        correlation_plots[plot_name] = plot_cross_correlation(
            patient_data, 'MR VC mm', col, patient_id
        )
    
    return correlation_plots

def generate_all_patient_plots(df: pd.DataFrame) -> Dict[str, bytes]:
    """
    Generate all correlation plots for all patients
    
    Returns:
    Dict[str, bytes]: Dictionary mapping plot names to plot images as bytes
    """
    all_plots = {}
    for patient_id in df['Patient ID'].unique():
        patient_plots = plot_patient_correlations(df, patient_id)
        all_plots.update(patient_plots)
    return all_plots