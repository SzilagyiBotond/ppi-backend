import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def load_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """The file should be in .xlsx format
    """
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        df_imputed = df.copy()

        # List of unique patient IDs
        patient_ids = df['Patient ID'].unique()

        # Perform imputation for each patient separately
        imputed_dfs = []
        for patient_id in patient_ids:
            patient_data = df[df['Patient ID'] == patient_id].copy()
            
            # Only perform imputation if there are missing values
            if patient_data.isnull().any().any():
                # Calculate appropriate number of neighbors (should be less than number of samples)
                n_neighbors = min(3, len(patient_data)-1)
                imputed_patient_data = impute_by_patient(patient_data, n_neighbors=n_neighbors)
                imputed_dfs.append(imputed_patient_data)
            else:
                imputed_dfs.append(patient_data)

        # Combine all imputed data
        df_imputed = pd.concat(imputed_dfs, axis=0).sort_index()
    
    df = df_imputed
    if 'Time (moment)' in df.columns:
        df['Time (moment)'] = pd.to_datetime(df['Time (moment)'], unit='ms')
    return df

def split_by_patient(df:pd.DataFrame) -> dict[str,pd.DataFrame]:
    patient_dfs = {}

    # Split the dataframe by Patient ID
    for patient_id in df['Patient ID'].unique():
        patient_dfs[patient_id] = df[df['Patient ID'] == patient_id].copy()

    return patient_dfs


def impute_by_patient(patient_data, n_neighbors=3):
    # Remove the ID and Cycle columns for imputation
    cols_to_impute = patient_data.columns.difference(['Patient ID', 'Cycle'])
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(patient_data[cols_to_impute])
    scaled_df = pd.DataFrame(scaled_data, columns=cols_to_impute)
    
    # Perform KNN imputation
    imputer = KNNImputer(n_neighbors=min(n_neighbors, len(patient_data)-1))
    imputed_data = imputer.fit_transform(scaled_df)
    
    # Inverse transform the scaled data
    imputed_data = scaler.inverse_transform(imputed_data)
    
    # Create DataFrame with imputed values
    imputed_df = pd.DataFrame(imputed_data, columns=cols_to_impute)
    
    # Add back the ID and Cycle columns
    imputed_df['Patient ID'] = patient_data['Patient ID'].values
    imputed_df['Cycle'] = patient_data['Cycle'].values
    
    return imputed_df
